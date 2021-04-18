# TD: REST

## 1. First step to REST

| Header | Operation | Endpoint             | Request Body                               | Response Body        |
|:------:|:---------:|:--------------------:|:------------------------------------------:|:--------------------:|
|        | GET       | /vehicles            | -                                          | all vehicles         |
|        | GET       | /vehicles/id         | -                                          | specific vehicle/404 |
|        | POST      | /vehicles/create     | name, model year, type, gearbox, fuel type | created vehicle      |
|        | PUT       | /vehicles/update/id  | model year                                 | updated vehicle/404  |
|        | DELETE    | /vehicles/delete/id  | -                                          | all vehicles/404     |

## 2. Online REST API

1. Done
2. GET https://gorest.co.in/public-api/users?name=Vyctor
3. GET https://gorest.co.in/public-api/users?gender=Female
4. PUT https://gorest.co.in/public-api/users/57 { "name": "Budhil Silver" }
5. POSTMAN
    - Done
    - Done
    - Done
6. CURL
    - curl -i -H "Accept:application/json" -H "Content-Type:application/json" -XGET "https://gorest.co.in/public-api/users?name=Vyctor"
    - curl -i -H "Accept:application/json" -H "Content-Type:application/json" -XGET "https://gorest.co.in/public-api/users?gender=Female"
    - curl -i -H "Accept:application/json" -H "Content-Type:application/json" -H "Authorization: Bearer ACCESS-TOKEN" -XPATCH "https://gorest.co.in/public-api/users/66" -d '{"name":"Bala Silver", "email":"allasani.peddana@15ce.com", "status":"Active"}'
