from configs import Pairing_multi
from configs import Pairing_vbf_v3, Pairing_vbf_v3_onnx
from configs import Pairing_vbf
from configs import EventCate

workflows = {}
workflows["Pairing_multi"] = Pairing_multi
workflows["Pairing_vbf"] = Pairing_vbf_v3
workflows["Pairing_vbf_v3_onnx"] = Pairing_vbf_v3_onnx
workflows["EventCate"] = EventCate

__all__ = ["workflows"]