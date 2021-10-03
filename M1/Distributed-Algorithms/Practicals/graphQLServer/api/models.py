from main import db


class Vehicle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    manufacturer = db.Column(db.String)
    model = db.Column(db.String)
    modelyear = db.Column(db.String)
    vehicletype = db.Column(db.String)
    gearbox = db.Column(db.String)
    fueltype = db.Column(db.String)
    reserved = db.Column(db.Boolean, default=False)
    personid = db.Column(db.Integer)

    def to_dict(self):
        return {
            "id": self.id,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "modelyear": self.modelyear,
            "vehicletype": self.vehicletype,
            "gearbox": self.gearbox,
            "fueltype": self.fueltype,
            "reserved": self.reserved,
            "personid": self.personid
        }

class Person(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    lastname = db.Column(db.String)

    def to_dictPerson(self, ve):
        return {
            "id": self.id,
            "name": self.name,
            "lastname": self.lastname,
            "vehicles": ve
        }