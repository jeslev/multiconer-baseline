import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from log import logger
from utils.reader_utils import get_ner_reader, extract_spans, _assign_ner_tags

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# dictionary_categories = {
#     # Medical
#     "Disease": "A disease is an impairment of health or a condition of abnormal functioning. A disease is a medical term.",
#     "MedicalProcedure": "A medical procedure is employed by medical or dental practitioners. A medical procedure is a medical term.",
#     "AnatomicalStructure": "An anatomical structure is a particular complex anatomical part of a living thing and its construction and arrangement. An anatomical structure is a medical term.",
#     "Medication/Vaccine": "A medication or vaccine is an immunogen consisting of a suspension of weakened or dead pathogenic cells injected in order to stimulate the production of antibodies. A medication or vacine is a medical term.",
#     "Symptom": "A symptom is any sensation or change in bodily function that is experienced by a patient and is associated with a particular disease. A symptom is a medical term.",
#     # Location
#     "Facility": "A facility is a building or place that provides a particular service or is used for a particular industry. A facility is a location.",
#     "HumanSettlement": "A human settlement is community of people smaller than a town. A human settlement is a location.",
#     "Station": "A station is a facility equipped with special equipment and personnel for a particular purpose. A station is a location.",
#     "OtherLOC": "A location is a determination of the place where something is.",
#     # Person
#     "Politician": "A politician is a person active in party politics. A politician is a person.",
#     "SportsManager": "A sports manager is someone in charge of training an athlete or a team. A sports manager is a person.",
#     "Cleric": "A cleric is a clergyman or other person in religious orders. A cleric is a person.",
#     "Artist": "An artist is a person whose creative work shows sensitivity and imagination. An artist is a person.",
#     "Athlete": "An athlete is a person trained to compete in sports. An athlete is a person.",
#     "OtherPER": "A person includes a legendary person whose existence is questionable, and a common name or nickname that constitutes the distinctive designation of a person.",
#     "Scientist": "A scientist is a person with advanced knowledge of one or more sciences. A scientist is a person.",
#     # Product
#     "Vehicle": "A vehicle is a conveyance that transports people or objects. A vehicle is a product.",
#     "Food": "The food is any substance that can be metabolized by an animal to give energy and build tissue. The food is a product.",
#     "Drink": "A drink is a single serving of a beverage. A drink is a product.",
#     "OtherPROD": "A product is the name of a thing or abstract concept produced by the intent of human beings.",
#     "Clothing": "A clothing is a covering designed to be worn on a person's body. A clothing is a product.",
#     # Creative Work
#     "MusicalWork": "A musical work is a play or film whose action and dialogue is interspersed with singing and dancing. A musical work is a creative work.",
#     "ArtWork": "An art work is photograph or other visual representation in a printed publication. An art work is a creative work.",
#     "WrittenWork": "A written work is something written by hand. A written work is a creative work.",
#     "VisualWork": "A visual work is something related to or using sigh. A visual work is a creative work.",
#     "Software": "A software is a written programs or procedures and associated documentation pertaining to the operation of a computer system. A software is a creative work.",
#     "OtherCW": "A creative work is a manifestation of creative effort including fine artwork, dance, writing , filmmaking, and composition.",
#     # Group
#     "PublicCorp": "A public corporation is a company, such as a national railway or a mail service, that is owned and managed by the government. A public corporation is a group.",
#     "ORG": "An organization is a group of people who work together. An organization is a group.",
#     "SportsGRP": "A sports group is a group of individuals who play sorts, usually team sports, on the same team. A sports group is a group.",
#     "CarManufacturer": "A car manufacturer is a business engaged in the manufacture of automobiles. A car manufacturer is a group.",
#     "AerospaceManufacturer": "An aerospace manufacturer is a company or individual involved in designing, building, testing, selling, and maintaining aircraft, missiles, rockets, or spacecraft. An aerospace manufacturer is a group.",
#     "PrivateCorp": "A private corporation is a company that is owned by one person or a small group of people, for example a family, and whose shares are not traded on a stock market. A private corporation is a group.",
#     "MusicalGRP": "A musical group is a group of people who perform instrumental and or vocal music, with the ensemble typically known by a distinct name. A musical group is a group.",
#     "OtherCORP": "A group is a number of persons or things that are located, gathered, or classed together.",
#     "TechCORP": "A technological corporation is a company that focuses primarily on the manufacturing, research and development of computing, telecommunication and consumer electronics. A techonlogical coporation is a group.",
# }

dictionary_categories = {
    # Location
    "LOC": "A location is a determination of the place where something is. There are different types of locations, such as facility, station or human settlement.",
    # Person
    "PER": "A person is a human being. A person includes a legendary person whose existence is questionable, and a common name or nickname that constitutes the distinctive designation of a person.",
    # Product
    "PROD": "A product is the name of a thing or abstract concept produced by the intent of human beings. A product also refers to commodities offered for sale.",
    # Creative Work
    "CW": "A creative work is a manifestation of creative effort including fine artwork (sculpture, paintings, drawing, sketching, performance art), dance, writing (literature), filmmaking, and composition.",
    # Group
    "GRP" : "A group is a number of persons or things that are located, gathered, or classed together. A group also refers to a cultural group, ethnic group, social group or an organization.",
    "CORP": "A corporation is an organization, usually a group of people or a company, authorized by the state to act as a single entity (a legal entity recognized by private and public law; a legal person in legal context) and recognized as such in law for certain purposes.",
}

dictionary_categories = {
    # Location
    "LOC": "Una localización es la determinación del lugar donde se encuentra algo. Hay distintos tipos de ubicaciones, como instalaciones, estaciones o asentamientos humanos.",
    # Person
    "PER": "Una persona es un ser humano. Una persona incluye una persona legendaria cuya existencia es cuestionable, y un nombre común o apodo que constituye la designación distintiva de una persona.",
    # Product
    "PROD": "Un producto es el nombre de una cosa o concepto abstracto producido por la intención del ser humano. Un producto también se refiere a las mercancías puestas a la venta.",
    # Creative Work
    "CW": "Una obra creativa es una manifestación de esfuerzo creativo que incluye las bellas artes (escultura, pintura, dibujo, esbozo, arte escénico), la danza, la escritura (literatura), el cine y la composición.",
    # Group
    "GRP" : "Un grupo es un número de personas o cosas que se encuentran, reúnen o clasifican juntas. Un grupo también se refiere a un grupo cultural, un grupo étnico, un grupo social o una organización.",
    "CORP": "Una sociedad anónima es una organización, normalmente un grupo de personas o una empresa, autorizada por el Estado para actuar como una entidad única (una persona jurídica reconocida por el derecho privado y público; una persona jurídica en el contexto legal) y reconocida como tal por la ley a determinados efectos.",
}



dictionary_categories_es = {
    # Medical
    "Disease": "Una enfermedad es una alteración de la salud o una condición de funcionamiento anormal. Una enfermedad es un término médico.",
    "MedicalProcedure": "Un procedimiento médico es empleado por médicos u odontólogos. Un procedimiento médico es un término médico.",
    "AnatomicalStructure": "Una estructura anatómica es una parte anatómica compleja particular de un ser vivo, así como su construcción y disposición. Una estructura anatómica es un término médico.",
    "Medication/Vaccine": "Un medicamento o vacuna es un inmunógeno consistente en una suspensión de células patógenas debilitadas o muertas que se inyecta para estimular la producción de anticuerpos. Un medicamento o vacuna es un término médico.",
    "Symptom": "Un síntoma es cualquier sensación o cambio en la función corporal que experimenta un paciente y que se asocia a una enfermedad concreta. Un síntoma es un término médico.",
    # Location
    "Facility": "Una instalación es un edificio o lugar que presta un servicio concreto o se utiliza para una industria determinada. Una instalación es un lugar.",
    "HumanSettlement": "Un asentamiento humano es una comunidad de personas más pequeña que una ciudad. Un asentamiento humano es un lugar.",
    "Station": "Una estación es una instalación dotada de equipos y personal especiales para un fin determinado. Una estación es un lugar.",
    "OtherLOC": "Una localización es la determinación del lugar donde se encuentra algo.",
    # Person
    "Politician": "Un político es una persona activa en la política de partidos. Un político es una persona.",
    "SportsManager": "Un director deportivo es la persona encargada de entrenar a un deportista o a un equipo. Un director deportivo es una persona.",
    "Cleric": "Un clérigo es un miembro del clero u otra persona perteneciente a órdenes religiosas. Un clérigo es una persona.",
    "Artist": "Un artista es una persona cuya obra creativa demuestra sensibilidad e imaginación. Un artista es una persona.",
    "Athlete": "Un atleta es una persona entrenada para competir en deportes. Un atleta es una persona.",
    "OtherPER": "Una persona incluye una persona legendaria cuya existencia es cuestionable, y un nombre común o apodo que constituye la designación distintiva de una persona.",
    "Scientist": "Un científico es una persona con conocimientos avanzados en una o varias ciencias. Un científico es una persona.",
    # Product
    "Vehicle": "Un vehículo es un medio de transporte de personas u objetos. Un vehículo es un producto.",
    "Food": "El alimento es cualquier sustancia que pueda ser metabolizada por un animal para darle energía y construir tejidos. El alimento es un producto.",
    "Drink": "Una bebida es una porción individual de una sustancia liquida digerible. Una bebida es un producto.",
    "OtherPROD": "Un producto es el nombre de una cosa o concepto abstracto producido por la intención de seres humanos.",
    "Clothing": "Una prenda de vestir es una cubierta diseñada para ser llevada sobre el cuerpo de una persona. Una prenda de vestir es un producto.",
    # Creative Work
    "MusicalWork": "Una obra musical es una obra de teatro o una película cuya acción y diálogos se intercalan con cantos y bailes. Una obra musical es una obra creativa.",
    "ArtWork": "Una obra de arte es una fotografía u otra representación visual en una publicación impresa. Una obra de arte es un trabajo creativo.",
    "WrittenWork": "Una obra escrita es algo escrito a mano. Una obra escrita es una obra creativa.",
    "VisualWork": "Una obra visual es algo relacionado con los suspiros o que los utiliza. Una obra visual es una obra creativa.",
    "Software": "Un software es un programa o procedimiento escrito y la documentación asociada relativa al funcionamiento de un sistema informático. Un software es una obra creativa.",
    "OtherCW": "Una obra creativa es una manifestación de esfuerzo creativo que incluye las bellas artes, la danza, la escritura, el cine y la composición.",
    # Group
    "PublicCorp": "Una empresa pública es una compañía, como un ferrocarril nacional o un servicio de correos, que es propiedad del gobierno y está gestionada por él. Una empresa pública es un grupo cooperativo.",
    "ORG": "Una organización es un grupo de personas que trabajan juntas. Una organización es un grupo cooperativo.",
    "SportsGRP": "Una agrupación deportiva es un grupo de individuos que practican deportes, normalmente de equipo, en el mismo equipo. Una agrupación deportiva es un grupo cooperativo.",
    "CarManufacturer": "Un fabricante de automóviles es una empresa dedicada a la fabricación de automóviles. Un fabricante de automóviles es un grupo cooperativo.",
    "AerospaceManufacturer": "Un fabricante aeroespacial es una empresa o persona dedicada al diseño, construcción, pruebas, venta y mantenimiento de aviones, misiles, cohetes o naves espaciales. Un fabricante aeroespacial es un grupo cooperativo.",
    "PrivateCorp": "Una sociedad privada es una empresa propiedad de una persona o de un pequeño grupo de personas, por ejemplo una familia, y cuyas acciones no cotizan en bolsa. Una empresa privada es un grupo cooperativo.",
    "MusicalGRP": "Un grupo musical es un conjunto de personas que interpretan música instrumental y/o vocal, y que suele conocerse por un nombre distinto. Un grupo musical es un grupo cooperativo.",
    "OtherCORP": "Un grupo es un número de personas o cosas que se encuentran, reúnen o se asocian juntas.",
    "TechCORP": "Una corporación tecnológica es una empresa que se dedica principalmente a la fabricación, investigación y desarrollo de productos informáticos, de telecomunicaciones y electrónica de consumo. Una corporación tecnológica es un grupo cooperativo.",
}



dictionary_categories_fr = {
    # Medical
    "Disease": "Une maladie est une altération de la santé ou un état de fonctionnement anormal. Une maladie est un terme médical.",
    "MedicalProcedure": "Une procédure médicale est employée par des praticiens médicaux ou dentaires. Un acte médical est un terme médical.",
    "AnatomicalStructure": "Une structure anatomique est une partie anatomique complexe particulière d'un être vivant, ainsi que sa construction et sa disposition. Une structure anatomique est un terme médical.",
    "Medication/Vaccine": "Un médicament ou un vaccin est un immunogène constitué d'une suspension de cellules pathogènes affaiblies ou mortes injectées afin de stimuler la production d'anticorps. Un médicament ou un vaccin est un terme médical.",
    "Symptom": "Un symptôme est toute sensation ou modification d'une fonction corporelle ressentie par un patient et associée à une maladie particulière. Un symptôme est un terme médical.",
    # Location
    "Facility": "Une installation est un bâtiment ou un lieu qui fournit un service particulier ou est utilisé pour une industrie particulière. Une installation est un lieu.",
    "HumanSettlement": "Un établissement humain est une communauté de personnes plus petite qu'une ville. Un établissement humain est un lieu.",
    "Station": "Une station est une installation équipée d'un matériel et d'un personnel spécial dans un but précis. Une station est un lieu.",
    "OtherLOC": "Un lieu est une détermination de l'endroit où se trouve quelque chose.",
    # Person
    "Politician": "Un politicien est une personne active dans la politique du parti. Un politicien est une personne.",
    "SportsManager": "Un coach sportif est une personne chargée d'entraîner un athlète ou une équipe. Un coach sportif est une personne.",
    "Cleric": "Un clerc est un membre du clergé ou une autre personne appartenant à un ordre religieux. Un clerc est une personne.",
    "Artist": "Un artiste est une personne dont le travail créatif fait preuve de sensibilité et d'imagination. Un artiste est une personne.",
    "Athlete": "Un athlète est une personne entraînée pour participer à des compétitions sportives. Un athlète est une personne.",
    "OtherPER": "Une personne comprend une personne légendaire dont l'existence est douteuse, et un nom commun ou un surnom qui constitue la désignation distinctive d'une personne.",
    "Scientist": "Un scientifique est une personne ayant des connaissances avancées dans une ou plusieurs sciences. Un scientifique est une personne.",
    # Product
    "Vehicle": "Un véhicule est un moyen de transport qui transporte des personnes ou des objets. Un véhicule est un produit.",
    "Food": "L'aliment est toute substance qui peut être métabolisée par un animal pour donner de l'énergie et construire des tissus. L'aliment est un produit.",
    "Drink": "Une boisson est une portion individuelle d'un liquide digerable. Une boisson est un produit.",
    "OtherPROD": "Un produit est le nom d'une chose ou d'un concept abstrait produit par l'intention d'êtres humains.",
    "Clothing": "Un vêtement est une enveloppe destinée à être portée sur le corps d'une personne. Un vêtement est un produit.",
    # Creative Work
    "MusicalWork": "Une œuvre musicale est une pièce de théâtre ou un film dont l'action et les dialogues sont entrecoupés de chants et de danses. Une œuvre musicale est est un travail créatif.",
    "ArtWork": "Une œuvre d'art est une photographie ou toute autre représentation visuelle dans une publication imprimée. Une œuvre d'art est un travail créatif.",
    "WrittenWork": "Une œuvre écrite est une chose écrite à la main. Une œuvre écrite est un travail créatif.",
    "VisualWork": "Une œuvre visuelle est quelque chose en rapport avec le soupir ou qui l'utilise. Une œuvre visuelle est un travail créatif.",
    "Software": "Un logiciel est un programme ou une procédure écrite et la documentation associée concernant le fonctionnement d'un système informatique. Un logiciel est un travail créatif.",
    "OtherCW": "Un travail créatif est une manifestation de l'effort créatif, y compris les beaux-arts, la danse, l'écriture, le cinéma et la composition.",
    # Group
    "PublicCorp": "Une société publique est une entreprise, telle qu'un chemin de fer national ou un service postal, qui est détenue et gérée par le gouvernement. Une entreprise publique est un groupe.",
    "ORG": "Une organisation est un groupe de personnes qui travaillent ensemble. Une organisation est un groupe.",
    "SportsGRP": "Un groupe sportif est un groupe d'individus qui pratiquent des sortes, généralement des sports d'équipe, dans la même équipe. Un groupe sportif est un groupe.",
    "CarManufacturer": "Un constructeur automobile est une entreprise qui s'occupe de la fabrication d'automobiles. Un constructeur automobile est un groupe.",
    "AerospaceManufacturer": "Un constructeur aérospatial est une entreprise ou un individu impliqué dans la conception, la construction, les essais, la vente et l'entretien d'avions, de missiles, de fusées ou d'engins spatiaux. Un fabricant aérospatial est un groupe.",
    "PrivateCorp": "Une société privée est une entreprise qui appartient à une personne ou à un petit groupe de personnes, par exemple une famille, et dont les actions ne sont pas négociées sur un marché boursier. Une société privée est un groupe.",
    "MusicalGRP": "Un groupe musical est un groupe de personnes qui interprètent de la musique instrumentale et/ou vocale, l'ensemble étant généralement connu sous un nom distinct. Un groupe musical est un groupe.",
    "OtherCORP": "Un groupe est un nombre de personnes ou de choses qui sont situées, rassemblées ou classées ensemble.",
    "TechCORP": "Une société technologique est une entreprise qui se concentre principalement sur la fabrication, la recherche et le développement de l'informatique, des télécommunications et de l'électronique grand public. Une coopérative technologique est un groupe.",
}




dictionary_categories_pt = {
    # Medical
    "Disease": "Uma doença é um comprometimento ou uma condição de funcionamento anormal da saúde. Uma doença é um termo médico.",
    "MedicalProcedure": "Um procedimento médico é empregado por profissionais médicos ou odontológicos. Um procedimento médico é um termo médico.",
    "AnatomicalStructure": "Uma estrutura anatômica é uma parte anatômica particularmente complexa de um ser vivo e sua construção e disposição. Uma estrutura anatômica é um termo médico.",
    "Medication/Vaccine": "Um medicamento ou vacina é um imunógeno que consiste em uma suspensão de células patogênicas enfraquecidas ou mortas injetadas a fim de estimular a produção de anticorpos. Um medicamento ou vacina é um termo médico.",
    "Symptom": "Um sintoma é qualquer sensação ou mudança na função corporal que é experimentada por um paciente e está associada a uma determinada doença. Um sintoma é um termo médico.",
    # Location
    "Facility": "Uma instalação é um edifício ou local que presta um determinado serviço ou é utilizado para uma determinada indústria. Uma instalação é um lugar.",
    "HumanSettlement": "Um assentamento humano é uma comunidade de pessoas menor do que uma cidade. Um assentamento humano é um lugar.",
    "Station": "Uma estação é uma instalação equipada com equipamentos especiais e pessoal para um determinado fim. Uma estação é um lugar.",
    "OtherLOC": "Um lugar é uma determinação do local onde algo está.",
    # Person
    "Politician": "Um político é uma pessoa ativa na política partidária. Um político é uma pessoa.",
    "SportsManager": "Um treinador esportivo é alguém encarregado de treinar um atleta ou uma equipe. Um treinador esportivo é uma pessoa.",
    "Cleric": "Um clérigo é um membro do clero ou outra pessoa de ordens religiosas. Um clérigo é uma pessoa.",
    "Artist": "Um artista é uma pessoa cujo trabalho criativo mostra sensibilidade e imaginação. Um artista é uma pessoa.",
    "Athlete": "Um atleta é uma pessoa treinada para competir em esportes. Um atleta é uma pessoa.",
    "OtherPER": "Uma pessoa inclui uma pessoa lendária cuja existência é questionável, e um nome ou apelido comum que constitui a designação distinta de uma pessoa.",
    "Scientist": "Um cientista é uma pessoa com conhecimentos avançados de uma ou mais ciências. Um cientista é uma pessoa.",
    # Product
    "Vehicle": "Um veículo é um meio de transporte que transporta pessoas ou objetos. Um veículo é um produto.",
    "Food": "O alimento é qualquer substância que pode ser metabolizada por um animal para dar energia e construir tecidos. O alimento é um produto.",
    "Drink": "Uma bebida é uma única porção de uma substância líquida digerível. Uma bebida é um produto.",
    "OtherPROD": "Um produto é o nome de uma coisa ou conceito abstrato produzido pela intenção do ser humano.",
    "Clothing": "Uma roupa é um revestimento projetado para ser usado no corpo de uma pessoa. Uma roupa é um produto.",
    # Creative Work
    "MusicalWork": "Uma obra musical é uma peça de teatro ou filme cuja ação e diálogo são intercalados com canto e dança. Uma obra musical é uma obra criativa.",
    "ArtWork": "Uma obra de arte é uma fotografia ou outra representação visual em uma publicação impressa. Uma obra de arte é uma obra criativa.",
    "WrittenWork": "Um trabalho escrito é algo escrito à mão. Um trabalho escrito é uma obra criativa.",
    "VisualWork": "Um trabalho visual é algo relacionado a ou usando suspiro. Um trabalho visual é uma obra criativa.",
    "Software": "Um software é um programa ou procedimento escrito e documentação associada relativa à operação de um sistema de computador. Um software é uma obra criativa.",
    "OtherCW": "Uma obra criativa é uma manifestação de esforço criativo, incluindo obras de arte, dança, escrita, produção de filmes e composição.",
    # Group
    "PublicCorp": "Uma empresa pública é uma empresa, tal como uma ferrovia nacional ou um serviço de correio, que é de propriedade e administrada pelo governo. Uma empresa pública é um corporaçao.",
    "ORG": "Uma organização é um grupo de pessoas que trabalham juntas. Uma organização é um grupo.",
    "SportsGRP": "Um grupo esportivo é um grupo de indivíduos que jogam, geralmente, esportes de equipe, na mesma equipe. Um grupo esportivo é um corporaçao.",
    "CarManufacturer": "Um fabricante de automóveis é uma empresa que se dedica à fabricação de automóveis. Um fabricante de automóveis é um corporaçao.",
    "AerospaceManufacturer": "Um fabricante aeroespacial é uma empresa ou indivíduo envolvido no projeto, construção, teste, venda e manutenção de aeronaves, mísseis, foguetes ou naves espaciais. Um fabricante aeroespacial é um corporaçao.",
    "PrivateCorp": "Uma corporação privada é uma empresa que pertence a uma pessoa ou a um pequeno grupo de pessoas, por exemplo, uma família, e cujas ações não são negociadas em uma bolsa de valores. Uma corporação privada é um corporaçao.",
    "MusicalGRP": "Um grupo musical é um grupo de pessoas que executam música instrumental e/ou vocal, com o conjunto tipicamente conhecido por um nome distinto. Um grupo musical é um corporaçao.",
    "OtherCORP": "Um grupo é um número de pessoas ou coisas que estão localizadas, reunidas ou classificadas juntas.",
    "TechCORP": "Uma corporação tecnológica é uma empresa que se concentra principalmente na fabricação, pesquisa e desenvolvimento de computação, telecomunicação e eletrônica de consumo. Uma coporação tecnlógica é um corporaçao.",
}



class CoNLLReader(Dataset):
    def __init__(self, max_instances=-1, max_length=50, target_vocab=None, pretrained_dir='', encoder_model='xlm-roberta-large'):
        self._max_instances = max_instances
        self._max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_dir + encoder_model)

        self.pad_token = self.tokenizer.special_tokens_map['pad_token']
        self.pad_token_id = self.tokenizer.get_vocab()[self.pad_token]
        self.sep_token = self.tokenizer.special_tokens_map['sep_token']

        self.label_to_id = {} if target_vocab is None else target_vocab
        self.instances = []
        self.instances_definitions = []
        # Helpers to extract tag names
        #self.all_ner_tags = set()
        #self.all_ner_tags_names = set()

    def get_target_size(self):
        return len(set(self.label_to_id.values()))

    def get_target_vocab(self):
        return self.label_to_id

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def read_data(self, data, train=False):
        dataset_name = data if isinstance(data, str) else 'dataframe'
        logger.info('Reading file {}'.format(dataset_name))
        instance_idx = 0

        for fields, metadata in get_ner_reader(data=data):
            if self._max_instances != -1 and instance_idx > self._max_instances:
                break
            sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_, mask, gathered_ners= self.parse_line_for_ner(fields=fields)
            tokens_tensor = torch.tensor(tokens_sub_rep, dtype=torch.long)
            tag_tensor = torch.tensor(coded_ner_, dtype=torch.long).unsqueeze(0)
            token_masks_rep = torch.tensor(token_masks_rep)
            mask_rep = torch.tensor(mask)
            #print(tokens_tensor, mask_rep, token_masks_rep, gold_spans_, tag_tensor)
            #print(gathered_ners)#, fields)

            # Gather mentions and definitions
            instances_definitions = []
            if train:
                input_phrases = [" ".join(g[0])+". "+dictionary_categories.get(g[-1]) for g in gathered_ners]
                #input_phrases = [" ".join(g[0])+". "+dictionary_categories_es.get(g[-1]) for g in gathered_ners] # spanish
                
                instances_definitions = [None,None]
                if len(input_phrases)>0:
                    input_tokens = self.tokenizer(input_phrases, padding=True, return_tensors='pt')
                    positions_rep = [(g[1],g[2]) for g in gathered_ners]

                    # for (ini_id,fin_id), name in gold_spans_.items():
                    #     if name=='O':
                    #         continue
                    #     def_category = dictionary_categories.get(name,None)
                    #     if def_category is None:
                    #         print("Not definition found for category!")
                    instances_definitions = [input_tokens, positions_rep]
                self.instances.append((tokens_tensor, mask_rep, token_masks_rep, gold_spans_, tag_tensor, instances_definitions))
            else:
                #input_phrases = [" ".join(fields[0]+" .")]
                #input_tokens = self.tokenizer(input_phrases, padding=True, return_tensors='pt')
                #instances_definitions = [input_tokens, None]
                self.instances.append((tokens_tensor, mask_rep, token_masks_rep, gold_spans_, tag_tensor))#, instances_definitions))
            instance_idx += 1
        logger.info('Finished reading {:d} instances from file {}'.format(len(self.instances), dataset_name))

    def parse_line_for_ner(self, fields):
        tokens_, ner_tags = fields[0], fields[-1]
        sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep, mask, gathered_ners = self.parse_tokens_for_ner(tokens_, ner_tags)
        # Used to extract NER tags from Multiconer
        # for ner_tag in ner_tags:
        #     if ner_tag not in self.all_ner_tags_names:
        #         idx = len(self.all_ner_tags)
        #         self.all_ner_tags.add((ner_tag,idx))
        #         self.all_ner_tags_names.add(ner_tag)
        #print(sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep, mask)
        gold_spans_ = extract_spans(ner_tags_rep)
        coded_ner_ = [self.label_to_id[tag] if tag in self.label_to_id else self.label_to_id['O'] for tag in ner_tags_rep]

        return sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_, mask, gathered_ners

    def parse_tokens_for_ner(self, tokens_, ner_tags):
        sentence_str = ''
        tokens_sub_rep, ner_tags_rep = [self.pad_token_id], ['O']
        token_masks_rep = [False]

        gathered_ners = []
        cur_ner_text = []
        ini_pos, fin_pos = 0,0
        cur_tag = ""
        start = False
        for idx, token in enumerate(tokens_):
            if self._max_length != -1 and len(tokens_sub_rep) > self._max_length:
                break
            sentence_str += ' ' + ' '.join(self.tokenizer.tokenize(token.lower()))
            rep_ = self.tokenizer(token.lower())['input_ids']
            rep_ = rep_[1:-1]
            tokens_sub_rep.extend(rep_)

            # if we have a NER here, in the case of B, the first NER tag is the B tag, the rest are I tags.
            ner_tag = ner_tags[idx]
            tags, masks = _assign_ner_tags(ner_tag, rep_)

            # If starting ner, save position (start,end), and name string
            if ner_tag.startswith("B-") and not start:
                start = True
                ini_pos = len(ner_tags_rep)
                fin_pos = ini_pos+len(tags)
                cur_tag = ner_tag.replace("B-","")
                cur_ner_text.append(token)
            elif ner_tag.startswith("B-") and start: # 2 consecutives
                gathered_ners.append([cur_ner_text, ini_pos, fin_pos, cur_tag])
                start = True
                ini_pos = len(ner_tags_rep)
                fin_pos = ini_pos+len(tags)
                cur_tag = ner_tag.replace("B-","")
                cur_ner_text = [token]
            elif ner_tag.startswith("I-"):
                fin_pos += len(tags)
                cur_ner_text.append(token)
            elif ner_tag=="O" and start: # detected tag
                start = False
                gathered_ners.append([cur_ner_text, ini_pos, fin_pos, cur_tag])
                # Clean everything
                cur_ner_text=[]
                ini_pos, fin_pos = -1,-1
                cur_tag = "O"
            ner_tags_rep.extend(tags)
            token_masks_rep.extend(masks)

        if start: # if last token not processed
            start = False
            gathered_ners.append([cur_ner_text, ini_pos, fin_pos, cur_tag])

        tokens_sub_rep.append(self.pad_token_id)
        ner_tags_rep.append('O')
        token_masks_rep.append(False)
        mask = [True] * len(tokens_sub_rep)
        return sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep, mask, gathered_ners


class CoNLLUntokenizedReader(Dataset):
    def __init__(self, max_instances=-1, max_length=50, target_vocab=None):
        self._max_instances = max_instances
        self._max_length = max_length

        self.label_to_id = {} if target_vocab is None else target_vocab
        self.instances = []

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def read_data(self, data):
        dataset_name = data if isinstance(data, str) else 'dataframe'
        logger.info('Reading file {}'.format(dataset_name))
        instance_idx = 0

        for fields, metadata in get_ner_reader(data=data):
            if self._max_instances != -1 and instance_idx > self._max_instances:
                break
            sentence_str, tags, gold_spans = self.parse_line_for_ner(fields=fields)

            self.instances.append((sentence_str, tags, gold_spans))
            instance_idx += 1
        logger.info('Finished reading {:d} instances from file {}'.format(len(self.instances), dataset_name))

    def parse_line_for_ner(self, fields):
        tokens_, ner_tags = fields[0], fields[-1]
        sentence, tags = self.parse_tokens_for_ner(tokens_, ner_tags)
        gold_spans_ = extract_spans(tags)

        return sentence, tags, gold_spans_

    def parse_tokens_for_ner(self, tokens_, ner_tags):
        sentence_str = ''
        ner_tags_rep = []
        for idx, token in enumerate(tokens_):
            if self._max_length != -1 and len(ner_tags_rep) > self._max_length:
                break
            sentence_str += ' {}'.format(token)
            ner_tags_rep.append(ner_tags[idx])
        return sentence_str, ner_tags_rep
