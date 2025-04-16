# Details about the docker image

Prerequisites:

* Have docker installed and running
* Have docker-compose installed

# Launch Elasticsearch old Traceparts DB sample

Run the `docker-env-setup.sh` script to build the required docker images and create local data folder for docker volumes.

Once the images are built you can do a `docker-compose up` in this folder to run both Jena Fuseki and Elasticsearch Databases.

To setup the ES index and load some sample data, run the `es_setup.sh` script. (You need cURL)

Some notes:

* After loading the data, you need to wait a few minutes for ES to have the data ready before querying.

