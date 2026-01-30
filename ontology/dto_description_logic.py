class DLConcept:
    def __init__(self, name):
        self.name = name
 
class DLNot:
    def __init__(self, concept):
        self.concept = concept
 
class DLAnd:
    def __init__(self, left, right):
        self.left = left
        self.right = right
 
class DLOr:
    def __init__(self, left, right):
        self.left = left
        self.right = right
 
class DLSome:
    def __init__(self, role, concept):
        self.role = role
        self.concept = concept
 
class DLOnly:
    def __init__(self, role, concept):
        self.role = role
        self.concept = concept
 
class DescriptionLogicEvaluator:
    """
    Evaluates Description Logic expressions against a molecule's features.
    """
    def __init__(self, dto_ontology, feature_mapper):
        self.dto = dto_ontology
        self.mapper = feature_mapper
 
    def evaluate(self, dl_expression, molecule_features):
        """
        Return True/False for a given molecule based on DL expression.
        
        Args:
            dl_expression: A DL object (DLConcept, DLAnd, etc.)
            molecule_features: Dictionary of features (concepts, roles)
        """
        if isinstance(dl_expression, DLConcept):
            return self.mapper.match_concept(dl_expression.name, molecule_features)
 
        if isinstance(dl_expression, DLNot):
            return not self.evaluate(dl_expression.concept, molecule_features)
 
        if isinstance(dl_expression, DLAnd):
            return self.evaluate(dl_expression.left, molecule_features) and \
                   self.evaluate(dl_expression.right, molecule_features)
 
        if isinstance(dl_expression, DLOr):
            return self.evaluate(dl_expression.left, molecule_features) or \
                   self.evaluate(dl_expression.right, molecule_features)
 
        if isinstance(dl_expression, DLSome):
            return self.mapper.match_role_some(dl_expression.role, dl_expression.concept, molecule_features)
 
        if isinstance(dl_expression, DLOnly):
            return self.mapper.match_role_only(dl_expression.role, dl_expression.concept, molecule_features)
 
        raise ValueError(f"Unsupported DL expression type: {type(dl_expression)}")
