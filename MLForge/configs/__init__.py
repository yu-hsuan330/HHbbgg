from configs import Pairing_multi
from configs import Pairing_vbf
from configs import EventCate

workflows = {}
workflows["Pairing_multi"] = Pairing_multi
workflows["Pairing_vbf"] = Pairing_vbf
workflows["EventCate"] = EventCate

__all__ = ["workflows"]