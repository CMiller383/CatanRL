from .enums import Resource

class Hex:
    def __init__(self, hex_id: int, number: int, resource: Resource, center: tuple):
        self.hex_id = hex_id
        self.number = number
        self.resource = resource
        self.center = center  # (x, y) coordinates

    def __repr__(self):
        return f"Hex(id={self.hex_id}, number={self.number}, resource={self.resource}, center={self.center})"
