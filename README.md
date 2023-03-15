# Bird Species Classification using ResNet18 and Fine-tuning with Interpretability using LIME
This project is focused on training a deep learning model to classify 500 different species of birds. The model used in this project is ResNet18, which has been fine-tuned using transfer learning to improve its accuracy. Additionally, interpretability has been incorporated into the model using the LIME library.

## Dataset
The dataset used in this project is the CUB-200-2011 dataset, which contains 11,788 images of 200 bird species. Each species has around 30 images, making it a highly challenging dataset. The dataset has been split into 70% training, 15% validation, and 15% testing sets.

## Model Architecture
The ResNet18 model was used as the base model, which was pre-trained on the ImageNet dataset. The final layer of the model was replaced with a fully connected layer with 500 output units, representing the 500 bird species. The model was trained using the fine-tuning approach, where only the final layer's weights were updated during training. The model was trained for 10 epochs with a batch size of 128 and an initial learning rate of 0.001.

## Interpretability
Interpretability was incorporated into the model using the LIME (Local Interpretable Model-Agnostic Explanations) library. LIME was used to generate explanations for the model's predictions, allowing us to understand the model's decision-making process better. LIME generates a heatmap that highlights the most important parts of an image that the model used to make its prediction.

## Results
The trained model achieved a test loss of 0.198 and a test accuracy of 95.04%. The interpretability features of the model showed that the model was making its predictions based on specific features of the birds, such as the beak's shape, the color of the feathers, and the size of the bird.

### Classification Report
                               precision    recall  f1-score   support

              ABBOTTS BABBLER       1.00      0.80      0.89         5
                ABBOTTS BOOBY       0.50      0.20      0.29         5
   ABYSSINIAN GROUND HORNBILL       1.00      0.80      0.89         5
        AFRICAN CROWNED CRANE       1.00      1.00      1.00         5
       AFRICAN EMERALD CUCKOO       1.00      1.00      1.00         5
            AFRICAN FIREFINCH       1.00      0.80      0.89         5
       AFRICAN OYSTER CATCHER       1.00      0.80      0.89         5
        AFRICAN PIED HORNBILL       1.00      0.20      0.33         5
          AFRICAN PYGMY GOOSE       1.00      1.00      1.00         5
                    ALBATROSS       0.83      1.00      0.91         5
               ALBERTS TOWHEE       1.00      1.00      1.00         5
         ALEXANDRINE PARAKEET       1.00      1.00      1.00         5
                ALPINE CHOUGH       1.00      1.00      1.00         5
        ALTAMIRA YELLOWTHROAT       1.00      1.00      1.00         5
              AMERICAN AVOCET       1.00      1.00      1.00         5
             AMERICAN BITTERN       1.00      1.00      1.00         5
                AMERICAN COOT       1.00      1.00      1.00         5
            AMERICAN FLAMINGO       1.00      1.00      1.00         5
           AMERICAN GOLDFINCH       1.00      1.00      1.00         5
             AMERICAN KESTREL       0.71      1.00      0.83         5
               AMERICAN PIPIT       1.00      1.00      1.00         5
            AMERICAN REDSTART       1.00      1.00      1.00         5
               AMERICAN ROBIN       1.00      1.00      1.00         5
              AMERICAN WIGEON       1.00      1.00      1.00         5
            AMETHYST WOODSTAR       1.00      1.00      1.00         5
                 ANDEAN GOOSE       1.00      0.80      0.89         5
               ANDEAN LAPWING       1.00      1.00      1.00         5
                ANDEAN SISKIN       1.00      1.00      1.00         5
                      ANHINGA       1.00      1.00      1.00         5
                     ANIANIAU       1.00      1.00      1.00         5
            ANNAS HUMMINGBIRD       1.00      1.00      1.00         5
                      ANTBIRD       0.71      1.00      0.83         5
           ANTILLEAN EUPHONIA       1.00      0.60      0.75         5
                      APAPANE       0.83      1.00      0.91         5
                  APOSTLEBIRD       0.83      1.00      0.91         5
              ARARIPE MANAKIN       1.00      1.00      1.00         5
            ASHY STORM PETREL       1.00      0.40      0.57         5
              ASHY THRUSHBIRD       1.00      0.80      0.89         5
           ASIAN CRESTED IBIS       1.00      0.80      0.89         5
           ASIAN DOLLARD BIRD       1.00      1.00      1.00         5
                AUCKLAND SHAQ       1.00      0.80      0.89         5
            AUSTRAL CANASTERO       0.83      1.00      0.91         5
         AUSTRALASIAN FIGBIRD       1.00      1.00      1.00         5
                     AVADAVAT       1.00      0.80      0.89         5
             AZARAS SPINETAIL       1.00      1.00      1.00         5
         AZURE BREASTED PITTA       1.00      1.00      1.00         5
                    AZURE JAY       0.83      1.00      0.91         5
                AZURE TANAGER       1.00      0.60      0.75         5
                    AZURE TIT       1.00      1.00      1.00         5
                  BAIKAL TEAL       1.00      0.80      0.89         5
                   BALD EAGLE       1.00      1.00      1.00         5
                    BALD IBIS       1.00      0.80      0.89         5
                BALI STARLING       1.00      1.00      1.00         5
             BALTIMORE ORIOLE       1.00      1.00      1.00         5
                   BANANAQUIT       1.00      1.00      1.00         5
             BAND TAILED GUAN       1.00      1.00      1.00         5
             BANDED BROADBILL       1.00      0.80      0.89         5
                  BANDED PITA       1.00      1.00      1.00         5
                 BANDED STILT       0.71      1.00      0.83         5
            BAR-TAILED GODWIT       1.00      0.80      0.89         5
                     BARN OWL       0.80      0.80      0.80         5
                 BARN SWALLOW       1.00      1.00      1.00         5
              BARRED PUFFBIRD       0.83      1.00      0.91         5
            BARROWS GOLDENEYE       1.00      0.60      0.75         5
         BAY-BREASTED WARBLER       1.00      1.00      1.00         5
               BEARDED BARBET       0.83      1.00      0.91         5
             BEARDED BELLBIRD       1.00      1.00      1.00         5
             BEARDED REEDLING       1.00      1.00      1.00         5
            BELTED KINGFISHER       1.00      1.00      1.00         5
             BIRD OF PARADISE       1.00      1.00      1.00         5
   BLACK AND YELLOW BROADBILL       1.00      1.00      1.00         5
                   BLACK BAZA       1.00      1.00      1.00         5
                BLACK COCKATO       1.00      1.00      1.00         5
        BLACK FACED SPOONBILL       1.00      1.00      1.00         5
              BLACK FRANCOLIN       1.00      1.00      1.00         5
          BLACK HEADED CAIQUE       1.00      1.00      1.00         5
           BLACK NECKED STILT       1.00      0.60      0.75         5
                BLACK SKIMMER       1.00      0.80      0.89         5
                   BLACK SWAN       1.00      1.00      1.00         5
             BLACK TAIL CRAKE       1.00      0.80      0.89         5
       BLACK THROATED BUSHTIT       1.00      1.00      1.00         5
          BLACK THROATED HUET       1.00      0.80      0.89         5
       BLACK THROATED WARBLER       1.00      1.00      1.00         5
      BLACK VENTED SHEARWATER       0.83      1.00      0.91         5
                BLACK VULTURE       0.62      1.00      0.77         5
       BLACK-CAPPED CHICKADEE       1.00      1.00      1.00         5
           BLACK-NECKED GREBE       1.00      1.00      1.00         5
       BLACK-THROATED SPARROW       1.00      1.00      1.00         5
         BLACKBURNIAM WARBLER       1.00      1.00      1.00         5
    BLONDE CRESTED WOODPECKER       1.00      1.00      1.00         5
               BLOOD PHEASANT       1.00      1.00      1.00         5
                    BLUE COAU       1.00      1.00      1.00         5
                  BLUE DACNIS       0.80      0.80      0.80         5
        BLUE GRAY GNATCATCHER       0.31      0.80      0.44         5
                BLUE GROSBEAK       1.00      0.80      0.89         5
                  BLUE GROUSE       1.00      1.00      1.00         5
                   BLUE HERON       0.83      1.00      0.91         5
                 BLUE MALKOHA       1.00      1.00      1.00         5
       BLUE THROATED TOUCANET       1.00      1.00      1.00         5
                     BOBOLINK       1.00      1.00      1.00         5
          BORNEAN BRISTLEHEAD       1.00      1.00      1.00         5
             BORNEAN LEAFBIRD       1.00      1.00      1.00         5
             BORNEAN PHEASANT       1.00      1.00      1.00         5
             BRANDT CORMARANT       0.83      1.00      0.91         5
            BREWERS BLACKBIRD       1.00      1.00      1.00         5
                BROWN CREPPER       1.00      0.40      0.57         5
         BROWN HEADED COWBIRD       1.00      0.80      0.89         5
                  BROWN NOODY       1.00      1.00      1.00         5
               BROWN THRASHER       1.00      1.00      1.00         5
                   BUFFLEHEAD       1.00      1.00      1.00         5
             BULWERS PHEASANT       1.00      1.00      1.00         5
            BURCHELLS COURSER       0.80      0.80      0.80         5
                  BUSH TURKEY       1.00      1.00      1.00         5
           CAATINGA CACHOLOTE       1.00      1.00      1.00         5
                  CACTUS WREN       1.00      1.00      1.00         5
            CALIFORNIA CONDOR       1.00      1.00      1.00         5
              CALIFORNIA GULL       0.71      1.00      0.83         5
             CALIFORNIA QUAIL       1.00      1.00      1.00         5
                CAMPO FLICKER       1.00      1.00      1.00         5
                       CANARY       1.00      0.80      0.89         5
                   CANVASBACK       1.00      0.80      0.89         5
         CAPE GLOSSY STARLING       1.00      1.00      1.00         5
                CAPE LONGCLAW       1.00      1.00      1.00         5
             CAPE MAY WARBLER       1.00      1.00      1.00         5
             CAPE ROCK THRUSH       0.83      1.00      0.91         5
                 CAPPED HERON       0.83      1.00      0.91         5
                 CAPUCHINBIRD       1.00      1.00      1.00         5
            CARMINE BEE-EATER       1.00      1.00      1.00         5
                 CASPIAN TERN       1.00      1.00      1.00         5
                    CASSOWARY       1.00      1.00      1.00         5
                CEDAR WAXWING       1.00      1.00      1.00         5
             CERULEAN WARBLER       1.00      1.00      1.00         5
              CHARA DE COLLAR       1.00      1.00      1.00         5
              CHATTERING LORY       1.00      1.00      1.00         5
    CHESTNET BELLIED EUPHONIA       1.00      1.00      1.00         5
     CHINESE BAMBOO PARTRIDGE       1.00      1.00      1.00         5
           CHINESE POND HERON       1.00      1.00      1.00         5
             CHIPPING SPARROW       0.62      1.00      0.77         5
              CHUCAO TAPACULO       1.00      1.00      1.00         5
             CHUKAR PARTRIDGE       1.00      1.00      1.00         5
              CINNAMON ATTILA       1.00      1.00      1.00         5
          CINNAMON FLYCATCHER       1.00      1.00      1.00         5
                CINNAMON TEAL       1.00      1.00      1.00         5
                 CLARKS GREBE       1.00      0.80      0.89         5
            CLARKS NUTCRACKER       0.83      1.00      0.91         5
            COCK OF THE  ROCK       1.00      1.00      1.00         5
                     COCKATOO       1.00      1.00      1.00         5
             COLLARED ARACARI       1.00      1.00      1.00         5
       COLLARED CRESCENTCHEST       1.00      0.80      0.89         5
             COMMON FIRECREST       1.00      1.00      1.00         5
               COMMON GRACKLE       1.00      0.80      0.89         5
          COMMON HOUSE MARTIN       1.00      1.00      1.00         5
                  COMMON IORA       1.00      1.00      1.00         5
                  COMMON LOON       1.00      1.00      1.00         5
              COMMON POORWILL       0.83      1.00      0.91         5
              COMMON STARLING       1.00      1.00      1.00         5
        COPPERY TAILED COUCAL       1.00      1.00      1.00         5
                  CRAB PLOVER       1.00      1.00      1.00         5
                   CRANE HAWK       1.00      1.00      1.00         5
     CREAM COLORED WOODPECKER       1.00      1.00      1.00         5
               CRESTED AUKLET       1.00      1.00      1.00         5
             CRESTED CARACARA       1.00      1.00      1.00         5
                 CRESTED COUA       1.00      1.00      1.00         5
             CRESTED FIREBACK       0.83      1.00      0.91         5
           CRESTED KINGFISHER       1.00      1.00      1.00         5
             CRESTED NUTHATCH       1.00      1.00      1.00         5
           CRESTED OROPENDOLA       1.00      1.00      1.00         5
        CRESTED SERPENT EAGLE       1.00      0.60      0.75         5
            CRESTED SHRIKETIT       1.00      1.00      1.00         5
       CRESTED WOOD PARTRIDGE       1.00      1.00      1.00         5
                 CRIMSON CHAT       1.00      0.80      0.89         5
              CRIMSON SUNBIRD       1.00      1.00      1.00         5
                         CROW       0.80      0.80      0.80         5
               CROWNED PIGEON       1.00      1.00      1.00         5
                   CUBAN TODY       1.00      1.00      1.00         5
                 CUBAN TROGON       1.00      1.00      1.00         5
         CURL CRESTED ARACURI       1.00      1.00      1.00         5
             D-ARNAUDS BARBET       1.00      1.00      1.00         5
            DALMATIAN PELICAN       0.83      1.00      0.91         5
        DARJEELING WOODPECKER       1.00      0.60      0.75         5
              DARK EYED JUNCO       1.00      0.80      0.89         5
             DAURIAN REDSTART       1.00      1.00      1.00         5
             DEMOISELLE CRANE       1.00      0.80      0.89         5
          DOUBLE BARRED FINCH       1.00      1.00      1.00         5
     DOUBLE BRESTED CORMARANT       1.00      1.00      1.00         5
       DOUBLE EYED FIG PARROT       1.00      1.00      1.00         5
             DOWNY WOODPECKER       1.00      1.00      1.00         5
                   DUSKY LORY       1.00      0.80      0.89         5
                  DUSKY ROBIN       0.71      1.00      0.83         5
                   EARED PITA       1.00      1.00      1.00         5
             EASTERN BLUEBIRD       1.00      1.00      1.00         5
           EASTERN BLUEBONNET       1.00      1.00      1.00         5
        EASTERN GOLDEN WEAVER       1.00      1.00      1.00         5
           EASTERN MEADOWLARK       1.00      1.00      1.00         5
              EASTERN ROSELLA       1.00      1.00      1.00         5
                EASTERN TOWEE       1.00      1.00      1.00         5
        EASTERN WIP POOR WILL       1.00      1.00      1.00         5
         EASTERN YELLOW ROBIN       1.00      1.00      1.00         5
          ECUADORIAN HILLSTAR       1.00      1.00      1.00         5
               EGYPTIAN GOOSE       1.00      1.00      1.00         5
               ELEGANT TROGON       1.00      1.00      1.00         5
            ELLIOTS  PHEASANT       1.00      0.80      0.89         5
              EMERALD TANAGER       1.00      1.00      1.00         5
              EMPEROR PENGUIN       0.83      1.00      0.91         5
                          EMU       1.00      0.80      0.89         5
                 ENGGANO MYNA       1.00      1.00      1.00         5
           EURASIAN BULLFINCH       1.00      0.80      0.89         5
       EURASIAN GOLDEN ORIOLE       0.83      1.00      0.91         5
              EURASIAN MAGPIE       1.00      1.00      1.00         5
           EUROPEAN GOLDFINCH       1.00      0.80      0.89         5
         EUROPEAN TURTLE DOVE       1.00      1.00      1.00         5
             EVENING GROSBEAK       1.00      0.80      0.89         5
               FAIRY BLUEBIRD       1.00      1.00      1.00         5
                FAIRY PENGUIN       0.83      1.00      0.91         5
                   FAIRY TERN       1.00      0.60      0.75         5
             FAN TAILED WIDOW       1.00      1.00      1.00         5
               FASCIATED WREN       0.83      1.00      0.91         5
                FIERY MINIVET       1.00      1.00      1.00         5
            FIORDLAND PENGUIN       1.00      1.00      1.00         5
        FIRE TAILLED MYZORNIS       1.00      1.00      1.00         5
              FLAME BOWERBIRD       1.00      1.00      1.00         5
                FLAME TANAGER       1.00      0.80      0.89         5
                      FRIGATE       1.00      1.00      1.00         5
            FRILL BACK PIGEON       1.00      1.00      1.00         5
                GAMBELS QUAIL       1.00      1.00      1.00         5
           GANG GANG COCKATOO       1.00      1.00      1.00         5
              GILA WOODPECKER       1.00      1.00      1.00         5
               GILDED FLICKER       1.00      1.00      1.00         5
                  GLOSSY IBIS       1.00      1.00      1.00         5
                 GO AWAY BIRD       0.83      1.00      0.91         5
            GOLD WING WARBLER       1.00      1.00      1.00         5
            GOLDEN BOWER BIRD       1.00      1.00      1.00         5
       GOLDEN CHEEKED WARBLER       1.00      1.00      1.00         5
          GOLDEN CHLOROPHONIA       1.00      1.00      1.00         5
                 GOLDEN EAGLE       1.00      1.00      1.00         5
              GOLDEN PARAKEET       0.71      1.00      0.83         5
              GOLDEN PHEASANT       1.00      1.00      1.00         5
                 GOLDEN PIPIT       1.00      1.00      1.00         5
               GOULDIAN FINCH       1.00      1.00      1.00         5
                     GRANDALA       1.00      1.00      1.00         5
                 GRAY CATBIRD       0.83      1.00      0.91         5
                GRAY KINGBIRD       0.71      1.00      0.83         5
               GRAY PARTRIDGE       1.00      1.00      1.00         5
                  GREAT ARGUS       0.80      0.80      0.80         5
               GREAT GRAY OWL       0.83      1.00      0.91         5
                GREAT JACAMAR       1.00      1.00      1.00         5
               GREAT KISKADEE       1.00      1.00      1.00         5
                  GREAT POTOO       0.80      0.80      0.80         5
                GREAT TINAMOU       0.83      1.00      0.91         5
                 GREAT XENOPS       1.00      1.00      1.00         5
                GREATER PEWEE       1.00      0.80      0.89         5
      GREATER PRAIRIE CHICKEN       1.00      1.00      1.00         5
          GREATOR SAGE GROUSE       1.00      1.00      1.00         5
              GREEN BROADBILL       1.00      1.00      1.00         5
                    GREEN JAY       1.00      1.00      1.00         5
                 GREEN MAGPIE       1.00      1.00      1.00         5
            GREEN WINGED DOVE       1.00      1.00      1.00         5
            GREY CUCKOOSHRIKE       1.00      1.00      1.00         5
       GREY HEADED FISH EAGLE       1.00      0.40      0.57         5
                  GREY PLOVER       1.00      1.00      1.00         5
            GROVED BILLED ANI       0.83      1.00      0.91         5
                GUINEA TURACO       1.00      1.00      1.00         5
                   GUINEAFOWL       1.00      1.00      1.00         5
                GURNEYS PITTA       1.00      1.00      1.00         5
                    GYRFALCON       1.00      1.00      1.00         5
                     HAMERKOP       1.00      1.00      1.00         5
               HARLEQUIN DUCK       1.00      1.00      1.00         5
              HARLEQUIN QUAIL       1.00      1.00      1.00         5
                  HARPY EAGLE       1.00      1.00      1.00         5
               HAWAIIAN GOOSE       1.00      1.00      1.00         5
                     HAWFINCH       1.00      1.00      1.00         5
                 HELMET VANGA       1.00      1.00      1.00         5
              HEPATIC TANAGER       0.62      1.00      0.77         5
           HIMALAYAN BLUETAIL       1.00      1.00      1.00         5
              HIMALAYAN MONAL       1.00      1.00      1.00         5
                      HOATZIN       1.00      1.00      1.00         5
             HOODED MERGANSER       1.00      1.00      1.00         5
                      HOOPOES       1.00      1.00      1.00         5
                  HORNED GUAN       1.00      1.00      1.00         5
                  HORNED LARK       1.00      1.00      1.00         5
                HORNED SUNGEM       0.71      1.00      0.83         5
                  HOUSE FINCH       1.00      0.80      0.89         5
                HOUSE SPARROW       0.80      0.80      0.80         5
               HYACINTH MACAW       1.00      1.00      1.00         5
               IBERIAN MAGPIE       0.83      1.00      0.91         5
                     IBISBILL       1.00      1.00      1.00         5
                IMPERIAL SHAQ       1.00      0.80      0.89         5
                    INCA TERN       1.00      1.00      1.00         5
               INDIAN BUSTARD       0.83      1.00      0.91         5
                 INDIAN PITTA       1.00      1.00      1.00         5
                INDIAN ROLLER       1.00      1.00      1.00         5
               INDIAN VULTURE       1.00      1.00      1.00         5
               INDIGO BUNTING       1.00      1.00      1.00         5
            INDIGO FLYCATCHER       1.00      0.80      0.89         5
              INLAND DOTTEREL       1.00      0.80      0.89         5
         IVORY BILLED ARACARI       1.00      1.00      1.00         5
                   IVORY GULL       0.71      1.00      0.83         5
                          IWI       1.00      0.80      0.89         5
                       JABIRU       0.83      1.00      0.91         5
                   JACK SNIPE       1.00      1.00      1.00         5
               JACOBIN PIGEON       1.00      0.60      0.75         5
             JANDAYA PARAKEET       1.00      1.00      1.00         5
               JAPANESE ROBIN       1.00      1.00      1.00         5
                 JAVA SPARROW       1.00      1.00      1.00         5
            JOCOTOCO ANTPITTA       1.00      1.00      1.00         5
                         KAGU       1.00      0.80      0.89         5
                       KAKAPO       1.00      1.00      1.00         5
                     KILLDEAR       1.00      1.00      1.00         5
                   KING EIDER       1.00      1.00      1.00         5
                 KING VULTURE       1.00      1.00      1.00         5
                         KIWI       1.00      1.00      1.00         5
                   KOOKABURRA       1.00      0.80      0.89         5
                 LARK BUNTING       1.00      1.00      1.00         5
                LAUGHING GULL       1.00      1.00      1.00         5
               LAZULI BUNTING       1.00      1.00      1.00         5
              LESSER ADJUTANT       1.00      1.00      1.00         5
                 LILAC ROLLER       1.00      1.00      1.00         5
                      LIMPKIN       1.00      1.00      1.00         5
                   LITTLE AUK       0.83      1.00      0.91         5
            LOGGERHEAD SHRIKE       1.00      0.80      0.89         5
               LONG-EARED OWL       0.62      1.00      0.77         5
                 LOONEY BIRDS       0.83      1.00      0.91         5
          LUCIFER HUMMINGBIRD       1.00      1.00      1.00         5
                 MAGPIE GOOSE       1.00      1.00      1.00         5
             MALABAR HORNBILL       0.45      1.00      0.62         5
         MALACHITE KINGFISHER       1.00      1.00      1.00         5
           MALAGASY WHITE EYE       1.00      1.00      1.00         5
                        MALEO       1.00      1.00      1.00         5
                 MALLARD DUCK       1.00      1.00      1.00         5
                 MANDRIN DUCK       1.00      1.00      1.00         5
              MANGROVE CUCKOO       1.00      0.80      0.89         5
                MARABOU STORK       1.00      1.00      1.00         5
              MASKED BOBWHITE       1.00      0.60      0.75         5
                 MASKED BOOBY       0.80      0.80      0.80         5
               MASKED LAPWING       1.00      0.80      0.89         5
               MCKAYS BUNTING       1.00      0.60      0.75         5
                       MERLIN       0.83      1.00      0.91         5
             MIKADO  PHEASANT       1.00      1.00      1.00         5
               MILITARY MACAW       1.00      1.00      1.00         5
                MOURNING DOVE       1.00      1.00      1.00         5
                         MYNA       1.00      1.00      1.00         5
               NICOBAR PIGEON       1.00      1.00      1.00         5
              NOISY FRIARBIRD       1.00      1.00      1.00         5
NORTHERN BEARDLESS TYRANNULET       1.00      1.00      1.00         5
            NORTHERN CARDINAL       1.00      1.00      1.00         5
             NORTHERN FLICKER       1.00      1.00      1.00         5
              NORTHERN FULMAR       0.71      1.00      0.83         5
              NORTHERN GANNET       0.83      1.00      0.91         5
             NORTHERN GOSHAWK       0.83      1.00      0.91         5
              NORTHERN JACANA       1.00      1.00      1.00         5
         NORTHERN MOCKINGBIRD       1.00      0.60      0.75         5
              NORTHERN PARULA       1.00      1.00      1.00         5
          NORTHERN RED BISHOP       0.83      1.00      0.91         5
            NORTHERN SHOVELER       1.00      1.00      1.00         5
             OCELLATED TURKEY       1.00      1.00      1.00         5
                 OKINAWA RAIL       1.00      1.00      1.00         5
       ORANGE BRESTED BUNTING       1.00      1.00      1.00         5
             ORIENTAL BAY OWL       1.00      1.00      1.00         5
            ORNATE HAWK EAGLE       1.00      0.60      0.75         5
                       OSPREY       0.71      1.00      0.83         5
                      OSTRICH       0.83      1.00      0.91         5
                     OVENBIRD       1.00      1.00      1.00         5
               OYSTER CATCHER       0.83      1.00      0.91         5
              PAINTED BUNTING       1.00      1.00      1.00         5
                       PALILA       1.00      1.00      1.00         5
             PALM NUT VULTURE       1.00      1.00      1.00         5
             PARADISE TANAGER       1.00      1.00      1.00         5
             PARAKETT  AKULET       1.00      1.00      1.00         5
                  PARUS MAJOR       1.00      1.00      1.00         5
      PATAGONIAN SIERRA FINCH       1.00      1.00      1.00         5
                      PEACOCK       1.00      0.80      0.89         5
             PEREGRINE FALCON       0.83      1.00      0.91         5
                  PHAINOPEPLA       1.00      1.00      1.00         5
             PHILIPPINE EAGLE       1.00      1.00      1.00         5
                   PINK ROBIN       1.00      1.00      1.00         5
            PLUSH CRESTED JAY       1.00      1.00      1.00         5
              POMARINE JAEGER       0.83      1.00      0.91         5
                       PUFFIN       1.00      1.00      1.00         5
                    PUNA TEAL       1.00      0.80      0.89         5
                 PURPLE FINCH       0.83      1.00      0.91         5
             PURPLE GALLINULE       1.00      1.00      1.00         5
                PURPLE MARTIN       1.00      1.00      1.00         5
              PURPLE SWAMPHEN       0.83      1.00      0.91         5
             PYGMY KINGFISHER       1.00      0.80      0.89         5
                  PYRRHULOXIA       1.00      0.80      0.89         5
                      QUETZAL       1.00      1.00      1.00         5
             RAINBOW LORIKEET       1.00      1.00      1.00         5
                    RAZORBILL       1.00      1.00      1.00         5
        RED BEARDED BEE EATER       1.00      1.00      1.00         5
            RED BELLIED PITTA       1.00      1.00      1.00         5
        RED BILLED TROPICBIRD       0.71      1.00      0.83         5
             RED BROWED FINCH       1.00      1.00      1.00         5
          RED FACED CORMORANT       1.00      1.00      1.00         5
            RED FACED WARBLER       1.00      1.00      1.00         5
                     RED FODY       1.00      1.00      1.00         5
              RED HEADED DUCK       0.83      1.00      0.91         5
        RED HEADED WOODPECKER       0.83      1.00      0.91         5
                     RED KNOT       1.00      1.00      1.00         5
      RED LEGGED HONEYCREEPER       0.80      0.80      0.80         5
             RED NAPED TROGON       1.00      1.00      1.00         5
          RED SHOULDERED HAWK       0.71      1.00      0.83         5
              RED TAILED HAWK       1.00      0.40      0.57         5
            RED TAILED THRUSH       1.00      1.00      1.00         5
         RED WINGED BLACKBIRD       0.83      1.00      0.91         5
          RED WISKERED BULBUL       1.00      1.00      1.00         5
             REGENT BOWERBIRD       1.00      1.00      1.00         5
         RING-NECKED PHEASANT       0.83      1.00      0.91         5
                   ROADRUNNER       1.00      1.00      1.00         5
                    ROCK DOVE       1.00      1.00      1.00         5
       ROSE BREASTED COCKATOO       1.00      1.00      1.00         5
       ROSE BREASTED GROSBEAK       1.00      1.00      1.00         5
            ROSEATE SPOONBILL       1.00      1.00      1.00         5
          ROSY FACED LOVEBIRD       1.00      1.00      1.00         5
            ROUGH LEG BUZZARD       0.62      1.00      0.77         5
             ROYAL FLYCATCHER       1.00      1.00      1.00         5
         RUBY CROWNED KINGLET       1.00      0.40      0.57         5
    RUBY THROATED HUMMINGBIRD       1.00      1.00      1.00         5
              RUDY KINGFISHER       1.00      1.00      1.00         5
            RUFOUS KINGFISHER       0.83      1.00      0.91         5
                RUFUOS MOTMOT       1.00      1.00      1.00         5
              SAMATRAN THRUSH       1.00      1.00      1.00         5
                  SAND MARTIN       1.00      1.00      1.00         5
               SANDHILL CRANE       0.83      1.00      0.91         5
               SATYR TRAGOPAN       1.00      1.00      1.00         5
                  SAYS PHOEBE       1.00      0.60      0.75         5
   SCARLET CROWNED FRUIT DOVE       1.00      1.00      1.00         5
      SCARLET FACED LIOCICHLA       1.00      1.00      1.00         5
                 SCARLET IBIS       1.00      1.00      1.00         5
                SCARLET MACAW       1.00      1.00      1.00         5
              SCARLET TANAGER       1.00      1.00      1.00         5
                     SHOEBILL       1.00      1.00      1.00         5
       SHORT BILLED DOWITCHER       1.00      1.00      1.00         5
              SMITHS LONGSPUR       1.00      1.00      1.00         5
                   SNOW GOOSE       1.00      1.00      1.00         5
                  SNOWY EGRET       1.00      1.00      1.00         5
                    SNOWY OWL       0.71      1.00      0.83         5
                 SNOWY PLOVER       1.00      1.00      1.00         5
                         SORA       1.00      1.00      1.00         5
             SPANGLED COTINGA       1.00      1.00      1.00         5
                SPLENDID WREN       1.00      1.00      1.00         5
        SPOON BILED SANDPIPER       0.80      0.80      0.80         5
              SPOTTED CATBIRD       1.00      1.00      1.00         5
       SPOTTED WHISTLING DUCK       1.00      0.80      0.89         5
        SRI LANKA BLUE MAGPIE       1.00      1.00      1.00         5
                 STEAMER DUCK       1.00      1.00      1.00         5
      STORK BILLED KINGFISHER       1.00      1.00      1.00         5
            STRIATED CARACARA       1.00      0.00      0.00         5
                  STRIPED OWL       0.83      1.00      0.91         5
             STRIPPED MANAKIN       1.00      1.00      1.00         5
             STRIPPED SWALLOW       1.00      1.00      1.00         5
                   SUNBITTERN       0.80      0.80      0.80         5
              SUPERB STARLING       1.00      1.00      1.00         5
                  SURF SCOTER       1.00      1.00      1.00         5
            SWINHOES PHEASANT       0.83      1.00      0.91         5
                   TAILORBIRD       1.00      1.00      1.00         5
                TAIWAN MAGPIE       1.00      1.00      1.00         5
                       TAKAHE       1.00      0.80      0.89         5
                TASMANIAN HEN       1.00      1.00      1.00         5
              TAWNY FROGMOUTH       1.00      0.60      0.75         5
                    TEAL DUCK       1.00      1.00      1.00         5
                    TIT MOUSE       1.00      1.00      1.00         5
                      TOUCHAN       1.00      1.00      1.00         5
            TOWNSENDS WARBLER       1.00      1.00      1.00         5
                 TREE SWALLOW       1.00      1.00      1.00         5
         TRICOLORED BLACKBIRD       1.00      0.80      0.89         5
            TROPICAL KINGBIRD       1.00      1.00      1.00         5
                TRUMPTER SWAN       1.00      1.00      1.00         5
               TURKEY VULTURE       0.83      1.00      0.91         5
             TURQUOISE MOTMOT       1.00      1.00      1.00         5
                UMBRELLA BIRD       1.00      1.00      1.00         5
                VARIED THRUSH       1.00      1.00      1.00         5
                        VEERY       1.00      1.00      1.00         5
         VENEZUELIAN TROUPIAL       1.00      1.00      1.00         5
                       VERDIN       1.00      1.00      1.00         5
          VERMILION FLYCATHER       1.00      1.00      1.00         5
      VICTORIA CROWNED PIGEON       1.00      1.00      1.00         5
       VIOLET BACKED STARLING       1.00      1.00      1.00         5
         VIOLET GREEN SWALLOW       1.00      1.00      1.00         5
                VIOLET TURACO       1.00      1.00      1.00         5
         VULTURINE GUINEAFOWL       1.00      1.00      1.00         5
                 WALL CREAPER       1.00      1.00      1.00         5
             WATTLED CURASSOW       1.00      1.00      1.00         5
              WATTLED LAPWING       0.83      1.00      0.91         5
                     WHIMBREL       0.62      1.00      0.77         5
           WHITE BROWED CRAKE       1.00      1.00      1.00         5
         WHITE CHEEKED TURACO       1.00      1.00      1.00         5
       WHITE CRESTED HORNBILL       1.00      0.60      0.75         5
      WHITE EARED HUMMINGBIRD       1.00      0.80      0.89         5
           WHITE NECKED RAVEN       1.00      1.00      1.00         5
          WHITE TAILED TROPIC       1.00      1.00      1.00         5
     WHITE THROATED BEE EATER       1.00      1.00      1.00         5
                  WILD TURKEY       1.00      1.00      1.00         5
             WILLOW PTARMIGAN       1.00      0.80      0.89         5
     WILSONS BIRD OF PARADISE       1.00      1.00      1.00         5
                    WOOD DUCK       0.83      1.00      0.91         5
                  WOOD THRUSH       1.00      1.00      1.00         5
                      WRENTIT       1.00      0.80      0.89         5
  YELLOW BELLIED FLOWERPECKER       1.00      1.00      1.00         5
               YELLOW CACIQUE       1.00      1.00      1.00         5
      YELLOW HEADED BLACKBIRD       1.00      1.00      1.00         5

                     accuracy                           0.95      2500
                    macro avg       0.96      0.95      0.95      2500
                 weighted avg       0.96      0.95      0.95      2500


## Files
- `notebooks/`: The folder containing Jupyter notebooks used in the project.
- `models/`: The folder containing the trained model weights and architecture. The best model is saved here.
- `README.md`: The readme file for the project.

## Conclusion
This project showcases the effectiveness of fine-tuning pre-trained models for challenging classification tasks. Additionally, incorporating interpretability into the model helps to understand the decision-making process better, which can be useful for debugging and improving the model.
