from housemodel.panels.HAN_frost_model import FrostModel
from housemodel.tools.new_configurator import load_config

class PVTPanel():
    def __init__(self, model=FrostModel):
        self.name = None
        self.model = model
        self.area  = None
        self.inclination = None
        self.azimuth = None

    @classmethod
    def from_yaml(cls, filename: str):
        pvt = cls()
        d = load_config(filename).get("PVT")
        pvt.name = d.get("name")
        pvt.area = d.get("area")
        pvt.inclination = d.get("incl_deg")
        pvt.azimuth = d.get("az_deg")
        return pvt