from main import db
from api.models import Vehicle, Person

db.create_all()
vehicle1 = Vehicle(manufacturer="Renault", 
                  model="Capture",
                  modelyear="2019", 
                  vehicletype="SUV", 
                  gearbox="Automatic", 
                  fueltype="Hybrid", 
                  reserved=False, 
                  personid=1)
vehicle2 = Vehicle(manufacturer="Peugeot", 
                  model="308", 
                  modelyear= "2020", 
                  vehicletype="Car", 
                  gearbox="Manual", 
                  fueltype="Gasoline", 
                  reserved=False, 
                  personid=1)
vehicle3 = Vehicle(manufacturer="Ford", 
                  model="Focus", 
                  modelyear= "2015", 
                  vehicletype="Car", 
                  gearbox="Manual", 
                  fueltype="Gasoline", 
                  reserved=False, 
                  personid=2)
vehicle4 = Vehicle(manufacturer="Renault", 
                  model="Clio", 
                  modelyear= "2021", 
                  vehicletype="Car", 
                  gearbox="Manual", 
                  fueltype="Gasoline", 
                  reserved=False, 
                  personid=1)
vehicle5 = Vehicle(manufacturer="Mercedes Benz", 
                  model="EQC", 
                  modelyear= "2019", 
                  vehicletype="SUV", 
                  gearbox="Automatic", 
                  fueltype="Hybrid", 
                  reserved=False, 
                  personid=2)

db.session.add(vehicle1)
db.session.add(vehicle2)
db.session.add(vehicle3)
db.session.add(vehicle4)
db.session.add(vehicle5)

person1 = Person(name="Tony",lastname="Stark")
person2 = Person(name="Bruce",lastname="Banner")

db.session.add(person1)
db.session.add(person2)
db.session.commit()