class DTOFeatureMapper:
    """
    Maps chemical features (Z-features) to DTO concepts and roles.
    """
    def __init__(self, dto_ontology):
        self.dto = dto_ontology
 
    def match_concept(self, concept_name, molecule_features):
        """
        Checks if the molecule belongs to a Concept.
        
        Args:
            concept_name: Name of the DTO concept (e.g., 'AromaticRing')
            molecule_features: Dict containing 'concepts' list
        """
        return concept_name in molecule_features.get("concepts", [])
 
    def match_role_some(self, role_name, concept, molecule_features):
        """
        Evaluates Existential Restriction: (role some Concept)
        e.g., hasFunctionalGroup some NitroGroup
        """
        role_values = molecule_features.get("roles", {}).get(role_name, [])
        # Check if any of the values for this role match the target concept
        # concept can be a DLConcept object or just a name string depending on usage
        target_name = concept.name if hasattr(concept, 'name') else concept
        return target_name in role_values
 
    def match_role_only(self, role_name, concept, molecule_features):
        """
        Evaluates Universal Restriction: (role only Concept)
        e.g., hasTarget only Kinase
        """
        role_values = molecule_features.get("roles", {}).get(role_name, [])
        if not role_values: return True # Vacuously true if no roles exist
        
        target_name = concept.name if hasattr(concept, 'name') else concept
        return all(rv == target_name for rv in role_values)
