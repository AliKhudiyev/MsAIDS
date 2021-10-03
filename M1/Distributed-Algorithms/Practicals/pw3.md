# GraphQL

## GraphQL API

1.
{
  ships {
    name
    home_port
    missions {
      flight
      name
    }
    year_built
    active
    image
  }
}

2.
{
        "name": "GO Pursuit",
        "home_port": "Port Canaveral",
        "missions": [
          {
            "flight": "60",
            "name": "TESS"
          },
          {
            "flight": "61",
            "name": "Bangabandhu-1"
          },
          {
            "flight": "63",
            "name": "SES-12"
          },
          {
            "flight": "65",
            "name": "Telstar 19V"
          }
        ],
        "year_built": 2007,
        "active": false,
        "image": "https://i.imgur.com/5w1ZWre.jpg"
}

3.
{
  rockets {
    name
    company
    description
    height {
      meters
    }
    first_flight
  }
}

4.
{
  *rocket1:* rocket(id: "falcon1") {
   company
   country
   name
   description
  }
  *rocket2:* rocket(id: "falcon9") {
   company
   country
   name
   description
   height {
    meters
   }
   diameter {
    meters
   }
  }
}

5.
{
  *rocket1:* rocket(id: "falcon1") {
   ...info
  }
  *rocket2:* rocket(id: "falcon9") {
   ...info
   height {
    meters
   }
   diameter {
    meters
   }
  }
}

fragment info on Rocket {
  company
  country
  name
  description
}

6. REST POST request