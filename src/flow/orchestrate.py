from pipeline.detection import detection_pipeline
from pipeline.classification import classification_pipeline

def detection_orchestrate(AUG_LIST, MODE):
    for instance in AUG_LIST:
        FILE_DIR = instance["source"]
        SAVE_DIR = instance["destination"]
        TRANSFORM = instance["transform"]
        MOSAIC_ALLOCATE = instance["mosaic_allocate"]
        MULTIPLIER = instance["multiplier"]

        detection_pipeline(file_dir=FILE_DIR,
                           save_dir=SAVE_DIR,
                           mode=MODE,
                           transform=TRANSFORM,
                           mosaic_allocate=MOSAIC_ALLOCATE,
                           multiplier=MULTIPLIER)
    
def classification_orchestrate(AUG_LIST, MODE):
    for instance in AUG_LIST:
        FILE_DIR = instance["source"]
        SAVE_DIR = instance["destination"]
        TRANSFORM = instance["transform"]
        MULTIPLIER = instance["multiplier"]

        classification_pipeline(file_dir=FILE_DIR,
                           save_dir=SAVE_DIR,
                           mode=MODE,
                           transform=TRANSFORM,
                           multiplier=MULTIPLIER)
