from fitting_code.ingestion import get_cubes_from_json
from fitting_code.ingestion import filtration


def ingest_all():
    get_cubes_from_json.download_cubes_from_json()
def select_and_ingest():
    filtration.run()
    get_cubes_from_json.download_cubes_from_json()
def select_all():
    filtration.run()