from owlready2 import get_ontology
import logging

logger = logging.getLogger(__name__)

class DTOOntology:
    """
    Wrapper for parsing Drug Target Ontology (DTO) using owlready2.
    """
    def __init__(self, ontology_path):
        logger.info(f"Loading ontology from {ontology_path}...")
        try:
            self.onto = get_ontology(ontology_path).load()
            self.classes = {}
            self.object_properties = {}
            self.annotation_properties = {}
            self._parse()
            logger.info("Ontology loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load ontology: {e}")
            self.onto = None

    def _parse(self):
        """Parses classes and properties into internal dictionaries."""
        if not self.onto: return

        for cls in self.onto.classes():
            self.classes[cls.name] = cls
 
        for prop in self.onto.object_properties():
            self.object_properties[prop.name] = prop
 
        for ann in self.onto.annotation_properties():
            self.annotation_properties[ann.name] = ann
 
    def get_class(self, name):
        return self.classes.get(name)
 
    def get_property(self, name):
        return self.object_properties.get(name)
 
    def get_annotation(self, name):
        return self.annotation_properties.get(name)
 
    def list_classes(self):
        return list(self.classes.keys())
 
    def list_properties(self):
        return list(self.object_properties.keys())
