from api import db
from .models import Vehicle, Person
from ariadne import convert_kwargs_to_snake_case

def resolve_vehicles(obj, info, **kwargs):
    try:
        vehicles = [vehicle.to_dict() for vehicle in Vehicle.query.all()]
        payload = {
            "success": True,
            "vehicles": vehicles
        }
    except Exception as error:
        payload = {
            "success": False,
            "errors": [str(error)]
        }
    return payload

@convert_kwargs_to_snake_case
def resolve_vehicle(obj, info, vehicle_id):
    try:
        vehicle = Vehicle.query.get(vehicle_id)
        payload = {
            "success": True,
            "vehicle": vehicle.to_dict()
        }
    except AttributeError:
        payload = {
            "success": False,
            "errors": [f"Vehicle with id {vehicle_id} not found"]
        }
    return payload

def resolve_persons(obj, info):
    try:
        persons = [person.to_dictPerson([vehicle.to_dict() for vehicle in Vehicle.query.all() if vehicle.personid == person.id]) for person in Person.query.all()]
        payload = {
            "success": True,
            "persons": persons
        }
    except Exception as error:
        payload = {
            "success": False,
            "errors": [str(error)]
        }
    return payload

@convert_kwargs_to_snake_case
def resolve_person(obj, info, person_id):
    try:
        person = Person.query.get(person_id)
        print(person)
        payload = {
            "success": True,
            "person": person.to_dictPerson([vehicle for vehicle in Vehicle.query.all() if vehicle.id == person_id])
        }
    except AttributeError:
        print('it is false :(')
        payload = {
            "success": False,
            "errors": [f'Person(id={person_id}) Not Found!']
        }
    return payload

@convert_kwargs_to_snake_case
def resolve_delete_vehicle(obj, info, vehicle_id):
    try:
        db.session.delete(Vehicle.query.get(vehicle_id))
        db.session.commit()
        payload = {
            "success": True
        }
    except AttributeError:
        payload = {
            "success": False,
            "errors": [f'Vehicle(id={vehicle_id}) Not Found!']
        }
    print(payload)
    return payload

@convert_kwargs_to_snake_case
def resolve_mark_reserved(obj, info, vehicle_id):
    try:
        vehicle = Vehicle.query.get(vehicle_id)
        vehicle.reserved = True

        db.session.add(vehicle)
        db.session.commit()

        payload = {
            "success": True
        }
    except AttributeError:
        payload = {
            "success": False,
            "errors": [f'Vehicle(id={vehicle_id}) Not Found!']
        }
    return payload

@convert_kwargs_to_snake_case
def resolve_update_vehicle(obj, info, vehicle_id, manufacturer, model, modelyear, vehicletype, gearbox, fueltype, personid):
    try:
        vehicle = Vehicle.query.get(vehicle_id)
        if vehicle:
            vehicle.manufacturer = manufacturer
            vehicle.model = model
            vehicle.modelyear = modelyear
            vehicle.vehicletype = vehicletype
            vehicle.gearbox = gearbox
            vehicle.fueltype = fueltype
            vehicle.personid = personid

        print(vehicle.model)

        db.sesssion.add(vehicle)
        db.session.commit()

        payload = {
            "success": True,
            "vehicle": vehicle.to_dict()
        }
    except AttributeError:
        payload = {
            "success": False,
            "errors": [f'Vehicle(id={vehicle_id}) Not Found!']
        }
    return payload
    