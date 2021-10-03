from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport

# Select your transport with a defined url endpoint
transport = AIOHTTPTransport(url="https://api.spacex.land/graphql/")

# Create a GraphQL client using the defined transport
client = Client(transport=transport, fetch_schema_from_transport=True)

# Provide a GraphQL query
query = gql(
    """
    query getContinents {
      continents {
        code
        name
      }
    }
"""
)

query5 = gql(
'''
{
  rocket1: rocket(id: "falcon1") {
    company
  }

  rocket2: rocket(id: "falcon9") {
    height {
      meters
    }
    diameter {
      meters
    }
  }
}
'''
)

# Execute the query on the transport
result = client.execute(query5)
print(result)
