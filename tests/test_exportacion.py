import tempfile
import unittest
from pathlib import Path

from entrenar_exportacion import CLASES, _obtener_grupo, _particionar_grupos
from yolo_inference import ExportacionClassifier


class ExportacionDecisionTests(unittest.TestCase):
    def test_internacional_tambien_es_apto_nacional(self):
        decision = ExportacionClassifier.decidir_categoria("Internacional", 0.91)
        self.assertEqual(decision["apto_nacional"], "Sí")
        self.assertEqual(decision["apto_internacional"], "Sí")

    def test_nacional_no_implica_internacional(self):
        decision = ExportacionClassifier.decidir_categoria("Nacional", 0.85)
        self.assertEqual(decision["apto_nacional"], "Sí")
        self.assertEqual(decision["apto_internacional"], "No")

    def test_etiqueta_exportacion_marca_ambos_mercados_aprobados(self):
        decision = ExportacionClassifier.decidir_categoria("Exportacion", 0.88)
        self.assertEqual(decision["apto_exportacion"], "Sí")
        self.assertEqual(decision["apto_nacional"], "Sí")
        self.assertEqual(decision["apto_internacional"], "Sí")

    def test_etiqueta_antigua_rechazo_es_no_apto_en_ambos_mercados(self):
        decision = ExportacionClassifier.decidir_categoria("Rechazo", 0.9)
        self.assertEqual(decision["apto_nacional"], "No")
        self.assertEqual(decision["apto_internacional"], "No")

    def test_baja_confianza_se_abstiene(self):
        decision = ExportacionClassifier.decidir_categoria("Internacional", 0.69)
        self.assertEqual(decision["clasificacion"], "No concluyente")
        self.assertEqual(decision["apto_nacional"], "No concluyente")
        self.assertEqual(decision["apto_internacional"], "No concluyente")

    def test_split_reproducible_y_sin_compartir_sesiones(self):
        with tempfile.TemporaryDirectory() as temp:
            imagenes = {
                clase: [
                    Path(temp) / clase / f"IMG_{fecha}_{clase}.jpg"
                    for fecha in ("20250101", "20250102", "20250103", "20250104")
                ]
                for clase in CLASES
            }
            primera = _particionar_grupos(imagenes, semilla=7)
            segunda = _particionar_grupos(imagenes, semilla=7)

        self.assertEqual(primera, segunda)
        sesiones_por_split = {
            split: {
                path.stem.split("_")[1]
                for _, path in elementos
            }
            for split, elementos in primera.items()
        }
        self.assertFalse(sesiones_por_split["train"] & sesiones_por_split["val"])
        for clase in CLASES:
            self.assertTrue(any(nombre == clase for nombre, _ in primera["train"]))
            self.assertTrue(any(nombre == clase for nombre, _ in primera["val"]))

    def test_lote_permanece_junto_aunque_tenga_fechas_distintas(self):
        primera = _obtener_grupo(Path("LOTE-ABC_IMG_20250101_fruta.jpg"))
        segunda = _obtener_grupo(Path("LOTE-ABC_IMG_20250102_fruta.jpg"))
        self.assertEqual(primera, segunda)


if __name__ == "__main__":
    unittest.main()
