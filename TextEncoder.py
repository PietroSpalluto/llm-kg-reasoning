import pandas as pd


class TextEncoder:
    """
    TextEncoder creates a prompt from a node features and neighbors. Only node of type 'drug' and
    'disease' have textual features.
    """
    def __init__(self, node_name: str, node_type: str, node_features: pd.Series, shortest_paths: list):
        """
        :param node_name: node name
        :param node_type: node type
        :param node_features: node features, empty if node type is not 'drug' or 'disease'
        :param shortest_paths: dictionary containing targets nodes as keys and a list of shortest paths
                               from the source node and each target
        """
        self.node_name = node_name
        self.node_type = node_type
        self.node_features = node_features
        self.shortest_paths = shortest_paths

        # dataframe containing a mapping which provides a natural language description of the relation
        self.relations_mapping = pd.read_csv("data/relations_mapping.csv")

    def encode(self) -> str:
        """
        Makes a natural language description of the node using its name, type, features and neighbors
        :return: a string containing the natural language description
        """
        # explain the node and its type
        head = '"{}" is a {}'.format(self.node_name, self.node_type)
        # select a function to compute textual node features
        if self.node_type == 'disease':
            textual_features = self.encode_disease_features()
        elif self.node_type == 'drug':
            textual_features = self.encode_drug_features()
        else:
            textual_features = ''

        textual_paths = self.encode_paths()
        paths_head = 'the following are some connection that "{}" has with other elements:\n'.format(self.node_name)
        textual_paths = '-' + '\n-'.join(textual_paths)
        textual_paths = paths_head + textual_paths

        textual_node = '\n'.join([head, textual_features, textual_paths]).strip()

        return textual_node

    def encode_paths(self) -> list:
        """
        Encodes all shortest paths in natural language, paths are in the form:
        head IS_TYPE head_type -> ... -> relation -> ... -> tail IS_TYPE tail type
        :return: a list of textual encoded paths
        """
        textual_paths = []
        for path in self.shortest_paths:
            # use a window to select a relation at a time
            window_left = 0
            window_right = 3
            path_split = path.split(' -> ')
            textual_relation = ''
            while window_right <= len(path_split):
                textual_relation = textual_relation + self.encode_relation(path_split[window_left:window_right],
                                                                           window_left)
                window_left += 2
                window_right += 2

            textual_paths.append(textual_relation)

        return textual_paths

    def encode_relation(self, relation: list, window_left: int) -> str:
        """
        Encode a single relation in natural language
        :param relation: a list of strings representing a relation in the form:
                         [head IS_TYPE head_type, relation, tail IS_TYPE tail type]
        :param window_left: used to check if the relation being encoded is not the first one in the path,
        in this case we do not add again the tail
        :return: encoded relation
        """
        head_name = relation[0].split(' IS_TYPE ')[0]
        head_type = relation[0].split(' IS_TYPE ')[1]
        rel = relation[1].split(' ')[0]
        display_relation = relation[1].split(' ')[1]
        tail_name = relation[2].split(' IS_TYPE ')[0]
        # tail_type = relation[2].split(' IS_TYPE ')[1]

        textual_relation = (
            self.relations_mapping[(self.relations_mapping['relation'] == rel) &
                                   (self.relations_mapping['display_relation'] == display_relation) &
                                   (self.relations_mapping['x_type'] == head_type)]
            ['mapping'].values)[0]

        if window_left == 0:
            textual_relation = 'The {} "{}" {}: {}'.format(head_type,
                                                           head_name,
                                                           textual_relation,
                                                           tail_name)
        else:
            textual_relation = ', "{}" {}: {}'.format(head_name,
                                                      textual_relation,
                                                      tail_name)

        return textual_relation

    def encode_disease_features(self) -> str:
        """
        Writes disease features in natural language and ignores missing values. Many disease
        have more than one row in the DataFrame so their features are joined into one string
        :return: a string representing the features
        """
        # some description of the disease
        head = 'the following are descriptions of "{}" from various sources:'.format(self.node_name)
        mondo_description = '\n'.join(self.make_list(self.node_features['mondo_name']))
        mondo_definition = '\n'.join(self.make_list(self.node_features['mondo_definition']))
        umls_description = '\n'.join(self.make_list(self.node_features['umls_description']))
        orphanet_definition = '\n'.join(self.make_list(self.node_features['orphanet_definition']))
        orphanet_clinical_description = '\n'.join(self.make_list(self.node_features['orphanet_clinical_description']))

        # generate a description of the disease
        description = '\n'.join([mondo_description, mondo_definition, umls_description,
                                 orphanet_definition, orphanet_clinical_description]).strip()

        description = self.check_length(head, description)

        # epidemiology information
        head = 'epidemiology information about "{}":'.format(self.node_name)
        orphanet_epidemiology = '\n'.join(self.make_list(self.node_features['orphanet_epidemiology']))

        epidemiology = self.check_length(head, orphanet_epidemiology)

        # management and treatment
        head = 'management and treatment of "{}":'.format(self.node_name)
        orphanet_management_and_treatment = '\n'.join(self.make_list(self.node_features['orphanet_management_and_treatment']))

        management_treatment = self.check_length(head, orphanet_management_and_treatment)

        # mayo information
        symptoms = 'symptoms of "{}":'.format(self.node_name)
        mayo_symptoms = '\n'.join(self.make_list(self.node_features['mayo_symptoms']))
        mayo_symptoms = self.check_length(symptoms, mayo_symptoms)

        causes = 'causes of "{}":'.format(self.node_name)
        mayo_causes = '\n'.join(self.make_list(self.node_features['mayo_causes']))
        mayo_causes = self.check_length(causes, mayo_causes)

        risk_factors = 'risk factors associated with "{}":'.format(self.node_name)
        mayo_risk_factors = '\n'.join(self.make_list(self.node_features['mayo_risk_factors']))
        mayo_risk_factors = self.check_length(risk_factors, mayo_risk_factors)

        complications = 'complications of "{}":'.format(self.node_name)
        mayo_complications = '\n'.join(self.make_list(self.node_features['mayo_complications']))
        mayo_complications = self.check_length(complications, mayo_complications)

        prevention = 'prevention of "{}":'.format(self.node_name)
        mayo_prevention = '\n'.join(self.make_list(self.node_features['mayo_prevention']))
        mayo_prevention = self.check_length(prevention, mayo_prevention)

        see_doc = 'when to see a doctor in case of "{}":'.format(self.node_name)
        mayo_see_doc = '\n'.join(self.make_list(self.node_features['mayo_see_doc']))
        mayo_see_doc = self.check_length(see_doc, mayo_see_doc)

        mayo_information = '\n'.join([mayo_symptoms, mayo_causes, mayo_risk_factors,
                                      mayo_complications, mayo_prevention, mayo_see_doc])

        # merge all the features and remove new lines
        textual_features = '\n'.join([description, epidemiology, management_treatment, mayo_information]).strip()

        return textual_features

    def encode_drug_features(self) -> str:
        # some description of the disease
        head = 'the following are descriptions of "{}" from various sources:'.format(self.node_name)
        description = '\n'.join(self.make_list(self.node_features['description']))

        description = self.check_length(head, description)

        # half life of the drug
        head = ('the half life is is the length of time required for the concentration of a particular substance '
                'to decrease to half of its starting dose in the body and for "{}" is:'.format(self.node_name))
        half_life = '\n'.join(self.make_list(self.node_features['half_life']))

        half_life = self.check_length(head, half_life)

        # indication
        head = 'how to use "{}":'.format(self.node_name)
        indication = '\n'.join(self.make_list(self.node_features['indication']))

        indication = self.check_length(head, indication)

        # mechanism of action
        head = 'mechanism of action of "{}":'.format(self.node_name)
        mechanism_of_action = '\n'.join(self.make_list(self.node_features['mechanism_of_action']))

        mechanism_of_action = self.check_length(head, mechanism_of_action)

        # protein binding
        head = 'how "{}" bounds with other proteins:'.format(self.node_name)
        protein_binding = '\n'.join(self.make_list(self.node_features['protein_binding']))

        protein_binding = self.check_length(head, protein_binding)

        # pharmacodynamics
        head = 'pharmacodynamics of "{}":'.format(self.node_name)
        pharmacodynamics = '\n'.join(self.make_list(self.node_features['pharmacodynamics']))

        pharmacodynamics = self.check_length(head, pharmacodynamics)

        # state
        head = 'state of the matter of "{}":'.format(self.node_name)
        state = '\n'.join(self.make_list(self.node_features['state']))

        state = self.check_length(head, state)

        # ATCs
        head = 'Level 1 ATC of "{}":'.format(self.node_name)
        atc_1 = '\n'.join(self.make_list(self.node_features['atc_1']))
        atc_1 = self.check_length(head, atc_1)
        head = 'Level 2 ATC of "{}":'.format(self.node_name)
        atc_2 = '\n'.join(self.make_list(self.node_features['atc_2']))
        atc_2 = self.check_length(head, atc_2)
        head = 'Level 3 ATC of "{}":'.format(self.node_name)
        atc_3 = '\n'.join(self.make_list(self.node_features['atc_3']))
        atc_3 = self.check_length(head, atc_3)
        head = 'Level 4 ATC of "{}":'.format(self.node_name)
        atc_4 = '\n'.join(self.make_list(self.node_features['atc_4']))
        atc_4 = self.check_length(head, atc_4)
        # merge all ATCs
        atcs = '\n'.join([atc_1, atc_2, atc_3, atc_4]).strip()

        atcs = self.check_length(head, atcs)

        # category
        head = '"{}":'.format(self.node_name)
        category = '\n'.join(self.make_list(self.node_features['category']))

        category = self.check_length(head, category)

        # group
        head = '"{}":'.format(self.node_name)
        group = '\n'.join(self.make_list(self.node_features['group']))

        group = self.check_length(head, group)

        # pathway
        head = '"{}":'.format(self.node_name)
        pathway = '\n'.join(self.make_list(self.node_features['pathway']))

        pathway = self.check_length(head, pathway)

        # molecular weight
        head = '"{}":'.format(self.node_name)
        molecular_weight = '\n'.join(self.make_list(self.node_features['molecular_weight']))

        molecular_weight = self.check_length(head, molecular_weight)

        # TPSA
        head = '"{}":'.format(self.node_name)
        tpsa = '\n'.join(self.make_list(self.node_features['tpsa']))

        tpsa = self.check_length(head, tpsa)

        # clogp
        head = '"{}":'.format(self.node_name)
        clogp = '\n'.join(self.make_list(self.node_features['clogp']))

        clogp = self.check_length(head, clogp)

        # merge all the features and remove new lines
        textual_features = '\n'.join([description, half_life, indication, mechanism_of_action,
                                      protein_binding, pharmacodynamics, state, atcs, category,
                                      group, pathway, molecular_weight, tpsa, clogp]).strip()

        return textual_features

    @staticmethod
    def check_length(head: str, string: str) -> str:
        """
        Check if the length of 'string' is greater than 0. If True returns head and string otherwise returns
        an empty string
        :param head: a string explaining the content of 'string'
        :param string: a string representing features
        :return: string
        """
        if len(string) > 0:
            return ('\n'.join([head, string])).strip()
        else:
            return ''

    @staticmethod
    def make_list(ser: pd.Series) -> list:
        """
        Converts a pandas Series in a list of string beginning with '-'
        :param ser: pandas Series containing natural language features
        :return: list of strings
        """
        return ['-' + e.strip() for e in ser.dropna().unique().tolist()]
