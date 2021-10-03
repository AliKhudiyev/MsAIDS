from api import app, db
from api import models
from ariadne import load_schema_from_path, make_executable_schema, \
    graphql_sync, snake_case_fallback_resolvers, ObjectType
from ariadne.constants import PLAYGROUND_HTML
from flask import request, jsonify
from api.queries import resolve_vehicles,       \
                        resolve_vehicle,        \
                        resolve_persons,        \
                        resolve_person,         \
                        resolve_delete_vehicle, \
                        resolve_mark_reserved,  \
                        resolve_update_vehicle

query = ObjectType("Query")
query.set_field("vehicles", resolve_vehicles)
query.set_field("vehicle", resolve_vehicle)
query.set_field("persons", resolve_persons)
query.set_field("person", resolve_person)

mutation = ObjectType("Mutation")
mutation.set_field("deleteVehicle", resolve_delete_vehicle)
mutation.set_field("markReserved", resolve_mark_reserved)
mutation.set_field("updateVehicle", resolve_update_vehicle)

type_defs = load_schema_from_path("schema.graphql")
schema = make_executable_schema(type_defs, query, mutation, snake_case_fallback_resolvers)


@app.route("/graphql", methods=["GET"])
def graphql_playground():
    return PLAYGROUND_HTML, 200


@app.route("/graphql", methods=["POST"])
def graphql_server():
    data = request.get_json()

    success, result = graphql_sync(
        schema,
        data,
        context_value=request,
        debug=app.debug
    )

    status_code = 200 if success else 400
    return jsonify(result), status_code