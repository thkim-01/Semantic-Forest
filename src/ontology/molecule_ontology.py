"""
Molecule Ontology: 화학 온톨로지 구조 정의
"""
from typing import List, Dict, Set
from owlready2 import *
import os


class MoleculeOntology:
    """화학 분자 온톨로지를 관리하는 클래스"""
    
    def __init__(self, ontology_path: str = "molecule_ontology.owl"):
        self.ontology_path = ontology_path
        self.onto = None
        self._create_ontology()
    
    def _create_ontology(self):
        """온톨로지 생성 및 기본 구조 정의"""
        self.onto = get_ontology("http://www.semanticweb.org/molecule/ontology")
        
        with self.onto:
            # Center Class 정의
            class Molecule(Thing):
                """중심 클래스: 분자"""
                pass
            
            # Subclasses for refinement
            class AromaticMolecule(Molecule):
                """방향족 분자"""
                pass
            
            class NonAromaticMolecule(Molecule):
                """비방향족 분자"""
                pass
            
            # --- New Ontology Structure for True SDT ---
            
            # 1. Define Substructure Class Hierarchy
            class Substructure(Thing):
                """화학적 부분 구조 (부모 클래스)"""
                pass

            class FunctionalGroup(Substructure):
                """작용기"""
                pass
            
            class RingSystem(Substructure):
                """고리 시스템"""
                pass
            
            # Specific Functional Groups (Subclasses)
            class Alcohol(FunctionalGroup): pass
            class Amine(FunctionalGroup): pass
            class Carboxyl(FunctionalGroup): pass
            class Carbonyl(FunctionalGroup): pass
            class Ether(FunctionalGroup): pass
            class Ester(FunctionalGroup): pass
            class Amide(FunctionalGroup): pass
            class Nitro(FunctionalGroup): pass
            class Halogen(FunctionalGroup): pass
            
            # Specific Ring Systems
            class BenzeneRing(RingSystem): pass
            class Heterocycle(RingSystem): pass

            # 2. Define Object Properties
            class hasSubstructure(Molecule >> Substructure):
                """Molecules have substructures"""
                pass
            
            class hasFunctionalGroupRel(hasSubstructure): # Sub-property
                """Relationship to functional groups"""
                range = [FunctionalGroup]
                pass

            class hasRingSystem(hasSubstructure): # Sub-property
                """Relationship to ring systems"""
                range = [RingSystem]
                pass

            # Data Properties (Legacy retained for hybrid approach)
            class hasMolecularWeight(Molecule >> float):
                pass
            
            class hasNumAtoms(Molecule >> int):
                pass
            
            class hasNumHeavyAtoms(Molecule >> int):
                pass
            
            class hasNumRotatableBonds(Molecule >> int):
                pass
            
            class hasNumHBA(Molecule >> int):
                """Hydrogen Bond Acceptor 수"""
                pass
            
            class hasNumHBD(Molecule >> int):
                """Hydrogen Bond Donor 수"""
                pass
            
            class hasNumRings(Molecule >> int):
                pass
            
            class hasNumAromaticRings(Molecule >> int):
                pass
            
            class hasAromaticity(Molecule >> bool):
                pass
            
            class hasLogP(Molecule >> float):
                pass
            
            class hasTPSA(Molecule >> float):
                """Topological Polar Surface Area"""
                pass
            
            class obeysLipinski(Molecule >> bool):
                pass
            
            class hasMWCategory(Molecule >> str):
                """Molecular Weight Category: Low, Medium, High"""
                pass
            
            class hasLogPCategory(Molecule >> str):
                """LogP Category: Hydrophilic, Moderate, Lipophilic"""
                pass
            
            class hasTPSACategory(Molecule >> str):
                """TPSA Category: Low, Medium, High"""
                pass
            
            # Legacy string property (optional to keep or remove if fully successfully replaced)
            class hasFunctionalGroup(Molecule >> str):
                """Functional Group 포함 여부 (Legacy String-based)"""
                pass
            
            class hasLabel(Molecule >> int):
                """Target label: 0 or 1"""
                pass
        
        self.Molecule = Molecule
        self.AromaticMolecule = AromaticMolecule
        self.NonAromaticMolecule = NonAromaticMolecule
        self.Substructure = Substructure
        self.FunctionalGroup = FunctionalGroup
        self.RingSystem = RingSystem
        self.hasSubstructure = hasSubstructure
        self.hasFunctionalGroupRel = hasFunctionalGroupRel
        self.hasRingSystem = hasRingSystem

        # Expose subclasses
        self.Alcohol = Alcohol
        self.Amine = Amine
        self.Carboxyl = Carboxyl
        self.Carbonyl = Carbonyl
        self.Ether = Ether
        self.Ester = Ester
        self.Amide = Amide
        self.Nitro = Nitro
        self.Halogen = Halogen
        self.BenzeneRing = BenzeneRing
        self.Heterocycle = Heterocycle
    
    def add_molecule_instance(self, mol_id: str, features: Dict, label: int):
        """분자 인스턴스를 온톨로지에 추가"""
        with self.onto:
            mol_instance = self.Molecule(mol_id)
            
            # Data properties 설정
            mol_instance.hasMolecularWeight = [features['molecular_weight']]
            mol_instance.hasNumAtoms = [features['num_atoms']]
            mol_instance.hasNumHeavyAtoms = [features['num_heavy_atoms']]
            mol_instance.hasNumRotatableBonds = [features['num_rotatable_bonds']]
            mol_instance.hasNumHBA = [features['num_hba']]
            mol_instance.hasNumHBD = [features['num_hbd']]
            mol_instance.hasNumRings = [features['num_rings']]
            mol_instance.hasNumAromaticRings = [features['num_aromatic_rings']]
            mol_instance.hasAromaticity = [features['has_aromatic']]
            mol_instance.hasLogP = [features['logp']]
            mol_instance.hasTPSA = [features['tpsa']]
            mol_instance.obeysLipinski = [features['obeys_lipinski']]
            mol_instance.hasMWCategory = [features['mw_category']]
            mol_instance.hasLogPCategory = [features['logp_category']]
            mol_instance.hasTPSACategory = [features['tpsa_category']]
            
            # Functional groups (Legacy)
            for fg in features['functional_groups']:
                mol_instance.hasFunctionalGroup.append(fg)
            
            # --- Populate Object Properties for True SDT ---
            # Map string features to Ontology Classes
            fg_map = {
                'Alcohol': self.Alcohol,
                'Amine': self.Amine,
                'Carboxyl': self.Carboxyl,
                'Carbonyl': self.Carbonyl,
                'Ether': self.Ether,
                'Ester': self.Ester,
                'Amide': self.Amide,
                'Nitro': self.Nitro,
                'Halogen': self.Halogen,
                'Benzene': self.BenzeneRing
            }

            for fg_name, fg_class in fg_map.items():
                # If the feature extractor detects this group (assuming features has boolean or count)
                # For now, we rely on the 'functional_groups' list which contains names
                if fg_name in features.get('functional_groups', []):
                    # Create an anonymous instance of the functional group
                    # or a specific one if we want to track unique groups.
                    # For SDT, existence is usually enough, so we create a distinct instance per molecule
                    fg_instance = fg_class()
                    mol_instance.hasFunctionalGroupRel.append(fg_instance)
                    
            # Aromatic Rings
            if features['has_aromatic']:
                mol_instance.is_a.append(self.AromaticMolecule)
            else:
                mol_instance.is_a.append(self.NonAromaticMolecule)

            # Label
            mol_instance.hasLabel = [label]
            
            return mol_instance
    
    def save(self):
        """온톨로지를 파일로 저장"""
        self.onto.save(file=self.ontology_path, format="rdfxml")
        print(f"Ontology saved to {self.ontology_path}")
    
    def load(self):
        """온톨로지를 파일에서 로드"""
        if os.path.exists(self.ontology_path):
            self.onto = get_ontology(self.ontology_path).load()
            print(f"Ontology loaded from {self.ontology_path}")
        else:
            print(f"Ontology file not found: {self.ontology_path}")
