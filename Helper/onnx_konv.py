import os
import json
import torch
import onnx
import onnxruntime
import gc
import random
import cv2
import numpy as np

# Zum Laden des Checkpoints via Ray
import ray.cloudpickle as pickle

# Quantisierung-Module von onnxruntime
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, CalibrationDataReader

# Für FP16-Konvertierung (optional)
try:
    from onnxconverter_common import float16
    HAS_FLOAT16_CONVERTER = True
except ImportError:
    HAS_FLOAT16_CONVERTER = False
    print("[WARN] float16_converter nicht verfügbar. FP16-Konvertierung wird übersprungen.")

class ONNXModelConverter:
    """
    Diese Klasse lädt ein PyTorch-Modell aus einem Checkpoint und ermöglicht den Export in das ONNX-Format
    in verschiedenen Präzisionen: FP32, FP16, INT8 (dynamisch und kalibriert).
    Alle nötigen Parameter werden bei der Initialisierung übergeben.
    """
    
    def __init__(self,
                 checkpoint_path: str,
                 model_name: str,
                 width: int = 2048,
                 height: int = 1024,
                 input_shape: tuple = (1, 3, 520, 520),
                 dynamic_batch: bool = True,
                 opset_version: int = 13,
                 output_dir: str = "./onnx_models",
                 skip_local_load: bool = True):
        """
        :param checkpoint_path: Pfad zum Checkpoint (pickle-Datei)
        :param model_name: Name des Modells (wird an TrainedModel übergeben)
        :param width: Eingabebreite des Modells (z. B. 2048)
        :param height: Eingabehöhe des Modells (z. B. 1024)
        :param input_shape: Form des Dummy-Inputs (z. B. (1,3,520,520))
        :param dynamic_batch: Ob die Batch-Dimension dynamisch sein soll
        :param opset_version: ONNX Opset Version für den Export
        :param output_dir: Verzeichnis, in dem die ONNX-Modelle gespeichert werden
        :param skip_local_load: Wird an den Modelldefinitions-Loader weitergereicht
        """
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.width = width
        self.height = height
        self.input_shape = input_shape
        self.dynamic_batch = dynamic_batch
        self.opset_version = opset_version
        self.output_dir = output_dir
        self.skip_local_load = skip_local_load
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None  # Wird später geladen

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        print(f"Initialisierung abgeschlossen. Gerät: {self.device}. Ausgabeordner: {self.output_dir}")

    def load_model(self):
        """
        Lädt das PyTorch-Modell aus dem angegebenen Checkpoint.
        Erwartet, dass in 'Helper/ml_models.py' die Klasse TrainedModel existiert.
        """
        try:
            from Helper.ml_models import TrainedModel
        except ImportError as e:
            raise ImportError("TrainedModel konnte nicht importiert werden. "
                              "Stelle sicher, dass Helper/ml_models.py im Pfad ist.") from e

        dummy_folder_path = "/tmp/onnx_export_temp"
        dummy_weights_name = "temp_weights"

        # Erzeuge das Model-Objekt (ohne lokale Gewichte zu laden)
        model_obj = TrainedModel(
            model_name=self.model_name,
            width=self.width,
            height=self.height,
            weights_name=dummy_weights_name,
            folder_path=dummy_folder_path,
            start_epoch="latest",
            skip_local_load=self.skip_local_load
        )

        # Prüfe, ob der Checkpoint existiert und lade ihn via pickle
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint nicht gefunden: {self.checkpoint_path}")

        with open(self.checkpoint_path, "rb") as fp:
            checkpoint_data = pickle.load(fp)
        
        # Wiederherstellen des state_dict
        model_obj.model.load_state_dict(checkpoint_data["model_state"])
        if "optimizer_state" in checkpoint_data:
            model_obj.optimizer.load_state_dict(checkpoint_data["optimizer_state"])
            
        model_obj.model.eval()
        model_obj.model.to(self.device)
        self.model = model_obj.model
        print(f"Modell {self.model_name} erfolgreich geladen und auf {self.device} platziert.")

    def export_fp32(self, onnx_filename: str = None):
        """
        Exportiert das geladene PyTorch-Modell als ONNX-Datei in FP32.
        :param onnx_filename: Dateiname für das exportierte Modell (optional)
        """
        if self.model is None:
            raise RuntimeError("Modell wurde noch nicht geladen. Bitte zuerst load_model() aufrufen.")

        if onnx_filename is None:
            onnx_filename = f"{self.model_name}_fp32.onnx"
        output_path = os.path.join(self.output_dir, onnx_filename)
        
        dummy_input = torch.randn(*self.input_shape, dtype=torch.float32, device=self.device)
        dynamic_axes = {}
        if self.dynamic_batch:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
            
        torch.onnx.export(
            model=self.model,
            args=dummy_input,
            f=output_path,
            export_params=True,
            opset_version=self.opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
        )
        print(f"FP32 ONNX-Modell exportiert: {output_path}")
        return output_path

    def convert_fp16(self, fp32_onnx_path: str, fp16_onnx_filename: str = None):
        """
        Konvertiert ein FP32 ONNX-Modell in FP16.
        :param fp32_onnx_path: Pfad zum FP32 ONNX-Modell
        :param fp16_onnx_filename: Dateiname für das FP16 Modell (optional)
        """
        if not os.path.isfile(fp32_onnx_path):
            raise FileNotFoundError(f"FP32 Modell nicht gefunden: {fp32_onnx_path}")
        if not HAS_FLOAT16_CONVERTER:
            raise ImportError("float16_converter nicht verfügbar. FP16-Konvertierung kann nicht durchgeführt werden.")
        
        if fp16_onnx_filename is None:
            base = os.path.splitext(os.path.basename(fp32_onnx_path))[0]
            fp16_onnx_filename = base + "_fp16.onnx"
        fp16_onnx_path = os.path.join(self.output_dir, fp16_onnx_filename)

        print(f"Starte FP16-Konvertierung: {fp32_onnx_path}")
        model = onnx.load(fp32_onnx_path)
        model_fp16 = float16.convert_float_to_float16(model)
        onnx.save(model_fp16, fp16_onnx_path)
        print(f"FP16 ONNX-Modell gespeichert: {fp16_onnx_path}")
        return fp16_onnx_path

    def quantize_int8_dynamic(self, fp32_onnx_path: str, int8_onnx_filename: str = None):
        """
        Führt eine dynamische INT8-Quantisierung durch.
        :param fp32_onnx_path: Pfad zum FP32 ONNX-Modell
        :param int8_onnx_filename: Dateiname für das INT8 Modell (optional)
        """
        if not os.path.isfile(fp32_onnx_path):
            raise FileNotFoundError(f"FP32 Modell nicht gefunden: {fp32_onnx_path}")
        
        if int8_onnx_filename is None:
            base = os.path.splitext(os.path.basename(fp32_onnx_path))[0]
            int8_onnx_filename = base + "_int8_dynamic.onnx"
        int8_onnx_path = os.path.join(self.output_dir, int8_onnx_filename)

        op_types_to_quantize = ['Conv', 'MatMul']
        print(f"Starte dynamische INT8-Quantisierung: {fp32_onnx_path}")
        quantize_dynamic(
            model_input=fp32_onnx_path,
            model_output=int8_onnx_path,
            op_types_to_quantize=op_types_to_quantize,
            weight_type=QuantType.QUInt8
        )
        print(f"Dynamisch quantisiertes INT8-Modell gespeichert: {int8_onnx_path}")
        return int8_onnx_path

    class CalibrationDataLoader(CalibrationDataReader):
        """
        Interner, benutzerdefinierter DataLoader für die kalibrierte INT8-Quantisierung.
        Nutzt eine Stichprobe von Bildern (max_samples) und optimiert den Speicherverbrauch.
        """
        def __init__(self, calibration_data_path: str, input_tensor_name: str, target_shape=(3, 520, 520), max_samples=300):
            self.calibration_data_path = calibration_data_path
            self.input_tensor_name = input_tensor_name
            self.target_shape = target_shape  # (Channels, Height, Width)
            self.max_samples = max_samples
            
            all_images = [os.path.join(calibration_data_path, f)
                          for f in os.listdir(calibration_data_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.image_paths = random.sample(all_images, min(len(all_images), self.max_samples))
            self.index = 0
            self.total_images = len(self.image_paths)
            
            if self.total_images == 0:
                raise ValueError("[ERROR] Keine Bilder für die Kalibrierung gefunden!")
            
            self.use_cuda = torch.cuda.is_available()
            print(f"Kalibrierungs-Dataloader: Nutze {self.total_images} Bilder. CUDA: {self.use_cuda}")

        def get_next(self):
            if self.index >= self.total_images:
                return None
            img_path = self.image_paths[self.index]
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Resize: Beachte, dass target_shape hier (C, H, W) ist, also Zielgröße (Width, Height) = (target_shape[2], target_shape[1])
            image = cv2.resize(image, (self.target_shape[2], self.target_shape[1]))
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            if self.use_cuda:
                image = torch.tensor(image).cuda()
            batch_image = np.expand_dims(image.cpu().numpy(), axis=0)
            del image
            self.index += 1
            torch.cuda.empty_cache()
            gc.collect()
            return {self.input_tensor_name: batch_image}

        def rewind(self):
            self.index = 0
            print("Kalibrierungs-Dataloader zurückgesetzt.")

    def quantize_int8_calibrated(self, fp32_onnx_path: str, calibration_data_path: str,
                                   input_tensor_name: str = "input", max_samples: int = 300,
                                   int8_onnx_filename: str = None):
        """
        Führt eine kalibrierte INT8-Quantisierung durch.
        :param fp32_onnx_path: Pfad zum FP32 ONNX-Modell
        :param calibration_data_path: Pfad zu den Kalibrierungsbildern
        :param input_tensor_name: Name des Eingabetensors im ONNX-Modell
        :param max_samples: Maximale Anzahl zu verwendender Bilder
        :param int8_onnx_filename: Dateiname für das quantisierte Modell (optional)
        """
        if not os.path.isfile(fp32_onnx_path):
            raise FileNotFoundError(f"FP32 Modell nicht gefunden: {fp32_onnx_path}")
        if int8_onnx_filename is None:
            base = os.path.splitext(os.path.basename(fp32_onnx_path))[0]
            int8_onnx_filename = base + "_int8_calibrated.onnx"
        int8_onnx_path = os.path.join(self.output_dir, int8_onnx_filename)
        
        data_reader = self.CalibrationDataLoader(
            calibration_data_path=calibration_data_path,
            input_tensor_name=input_tensor_name,
            target_shape=(3, self.input_shape[2], self.input_shape[3]) if len(self.input_shape) == 4 else (3,520,520),
            max_samples=max_samples
        )
        print(f"Starte kalibrierte INT8-Quantisierung: {fp32_onnx_path}")
        quantize_static(
            model_input=fp32_onnx_path,
            model_output=int8_onnx_path,
            calibration_data_reader=data_reader
        )
        print(f"Kalibrierte INT8-Quantisierung abgeschlossen: {int8_onnx_path}")
        return int8_onnx_path

    def test_model(self, onnx_model_path: str, test_input: np.ndarray = None):
        """
        Testet das ONNX-Modell, indem ein Inferenzlauf mit onnxruntime durchgeführt wird.
        :param onnx_model_path: Pfad zum ONNX-Modell
        :param test_input: Optional: Ein konkreter Testinput. Falls nicht angegeben, wird ein Dummy-Input verwendet.
        """
        if not os.path.isfile(onnx_model_path):
            raise FileNotFoundError(f"ONNX Modell nicht gefunden: {onnx_model_path}")
            
        session = onnxruntime.InferenceSession(onnx_model_path)
        input_name = session.get_inputs()[0].name
        
        # Erzeuge einen Dummy-Input, falls keiner übergeben wurde
        if test_input is None:
            test_input = np.random.randn(*self.input_shape).astype(np.float32)
        
        print(f"Teste Modell {onnx_model_path} mit Eingabeform {test_input.shape}")
        outputs = session.run(None, {input_name: test_input})
        print("Inferenz erfolgreich. Ausgabeformen:")
        for i, output in enumerate(outputs):
            print(f" - Output {i}: shape = {output.shape}")
        return outputs

# # Beispielhafte Nutzung:
# if __name__ == "__main__":
#     # Parameter anpassen!
#     converter = ONNXModelConverter(
#         checkpoint_path="/pfad/zum/checkpoint.pkl",
#         model_name="mein_modell",
#         width=2048,
#         height=1024,
#         input_shape=(1, 3, 520, 520),
#         dynamic_batch=True,
#         opset_version=13,
#         output_dir="./onnx_models",
#         skip_local_load=True
#     )

#     # Modell laden
#     converter.load_model()
    
#     # FP32-Export
#     fp32_path = converter.export_fp32()
    
#     # FP16-Konvertierung
#     if HAS_FLOAT16_CONVERTER:
#         fp16_path = converter.convert_fp16(fp32_path)
    
#     # Dynamische INT8-Quantisierung
#     int8_dyn_path = converter.quantize_int8_dynamic(fp32_path)
    
#     # Kalibrierte INT8-Quantisierung (Pfad zu Kalibrierungsbildern anpassen)
#     int8_cal_path = converter.quantize_int8_calibrated(
#         fp32_onnx_path=fp32_path,
#         calibration_data_path="/pfad/zu/kalibrierungsbildern",
#         input_tensor_name="input",
#         max_samples=300
#     )
    
#     # Teste die Modelle
#     print("\nTeste FP32 Modell:")
#     converter.test_model(fp32_path)
#     if HAS_FLOAT16_CONVERTER:
#         print("\nTeste FP16 Modell:")
#         converter.test_model(fp16_path)
#     print("\nTeste dynamisch quantisiertes INT8 Modell:")
#     converter.test_model(int8_dyn_path)
#     print("\nTeste kalibriertes INT8 Modell:")
#     converter.test_model(int8_cal_path)
