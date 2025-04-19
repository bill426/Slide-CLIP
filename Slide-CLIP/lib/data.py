# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py
import ssl
ssl._create_default_https_context=ssl._create_unverified_context
import  os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
import torch
from datasets import load_dataset
import torch
import os
from PIL import Image
import io
from torch.utils.data import  DataLoader

global processor

LC25000_Lung_classes_map={
    0:'0',
    1:'1',
    2:'2',
}

LC25000_Lung_templates = [
    'this is an image of "{}"',
]

LC25000_Colon_templates = [
    'a photo of "{}"',
]


svhn_classes_map={
    0:'0',
    1:'1',
    2:'2',
    3:'3',
    4:'4',
    5:'5',
    6:'6',
    7:'7',
    8:'8',
    9:'9'
}

svhn_templates = [
    'a photo of the number: "{}".',
]

mnist_classes_map={
    0:'0',
    1:'1',
    2:'2',
    3:'3',
    4:'4',
    5:'5',
    6:'6',
    7:'7',
    8:'8',
    9:'9'
}

mnist_templates = [
    'a photo of the number: "{}".',
]

cars_classes_map={
    0: 'AM General Hummer SUV 2000',
    1: 'Acura RL Sedan 2012',
    2: 'Acura TL Sedan 2012',
    3: 'Acura TL Type-S 2008',
    4: 'Acura TSX Sedan 2012',
    5: 'Acura Integra Type R 2001',
    6: 'Acura ZDX Hatchback 2012',
    7: 'Aston Martin V8 Vantage Convertible 2012',
    8: 'Aston Martin V8 Vantage Coupe 2012',
    9: 'Aston Martin Virage Convertible 2012',
    10: 'Aston Martin Virage Coupe 2012',
    11: 'Audi RS 4 Convertible 2008',
    12: 'Audi A5 Coupe 2012',
    13: 'Audi TTS Coupe 2012',
    14: 'Audi R8 Coupe 2012',
    15: 'Audi V8 Sedan 1994',
    16: 'Audi 100 Sedan 1994',
    17: 'Audi 100 Wagon 1994',
    18: 'Audi TT Hatchback 2011',
    19: 'Audi S6 Sedan 2011',
    20: 'Audi S5 Convertible 2012',
    21: 'Audi S5 Coupe 2012',
    22: 'Audi S4 Sedan 2012',
    23: 'Audi S4 Sedan 2007',
    24: 'Audi TT RS Coupe 2012',
    25: 'BMW ActiveHybrid 5 Sedan 2012',
    26: 'BMW 1 Series Convertible 2012',
    27: 'BMW 1 Series Coupe 2012',
    28: 'BMW 3 Series Sedan 2012',
    29: 'BMW 3 Series Wagon 2012',
    30: 'BMW 6 Series Convertible 2007',
    31: 'BMW X5 SUV 2007',
    32: 'BMW X6 SUV 2012',
    33: 'BMW M3 Coupe 2012',
    34: 'BMW M5 Sedan 2010',
    35: 'BMW M6 Convertible 2010',
    36: 'BMW X3 SUV 2012',
    37: 'BMW Z4 Convertible 2012',
    38: 'Bentley Continental Supersports Conv. Convertible 2012',
    39: 'Bentley Arnage Sedan 2009',
    40: 'Bentley Mulsanne Sedan 2011',
    41: 'Bentley Continental GT Coupe 2012',
    42: 'Bentley Continental GT Coupe 2007',
    43: 'Bentley Continental Flying Spur Sedan 2007',
    44: 'Bugatti Veyron 16.4 Convertible 2009',
    45: 'Bugatti Veyron 16.4 Coupe 2009',
    46: 'Buick Regal GS 2012',
    47: 'Buick Rainier SUV 2007',
    48: 'Buick Verano Sedan 2012',
    49: 'Buick Enclave SUV 2012',
    50: 'Cadillac CTS-V Sedan 2012',
    51: 'Cadillac SRX SUV 2012',
    52: 'Cadillac Escalade EXT Crew Cab 2007',
    53: 'Chevrolet Silverado 1500 Hybrid Crew Cab 2012',
    54: 'Chevrolet Corvette Convertible 2012',
    55: 'Chevrolet Corvette ZR1 2012',
    56: 'Chevrolet Corvette Ron Fellows Edition Z06 2007',
    57: 'Chevrolet Traverse SUV 2012',
    58: 'Chevrolet Camaro Convertible 2012',
    59: 'Chevrolet HHR SS 2010',
    60: 'Chevrolet Impala Sedan 2007',
    61: 'Chevrolet Tahoe Hybrid SUV 2012',
    62: 'Chevrolet Sonic Sedan 2012',
    63: 'Chevrolet Express Cargo Van 2007',
    64: 'Chevrolet Avalanche Crew Cab 2012',
    65: 'Chevrolet Cobalt SS 2010',
    66: 'Chevrolet Malibu Hybrid Sedan 2010',
    67: 'Chevrolet TrailBlazer SS 2009',
    68: 'Chevrolet Silverado 2500HD Regular Cab 2012',
    69: 'Chevrolet Silverado 1500 Classic Extended Cab 2007',
    70: 'Chevrolet Express Van 2007',
    71: 'Chevrolet Monte Carlo Coupe 2007',
    72: 'Chevrolet Malibu Sedan 2007',
    73: 'Chevrolet Silverado 1500 Extended Cab 2012',
    74: 'Chevrolet Silverado 1500 Regular Cab 2012',
    75: 'Chrysler Aspen SUV 2009',
    76: 'Chrysler Sebring Convertible 2010',
    77: 'Chrysler Town and Country Minivan 2012',
    78: 'Chrysler 300 SRT-8 2010',
    79: 'Chrysler Crossfire Convertible 2008',
    80: 'Chrysler PT Cruiser Convertible 2008',
    81: 'Daewoo Nubira Wagon 2002',
    82: 'Dodge Caliber Wagon 2012',
    83: 'Dodge Caliber Wagon 2007',
    84: 'Dodge Caravan Minivan 1997',
    85: 'Dodge Ram Pickup 3500 Crew Cab 2010',
    86: 'Dodge Ram Pickup 3500 Quad Cab 2009',
    87: 'Dodge Sprinter Cargo Van 2009',
    88: 'Dodge Journey SUV 2012',
    89: 'Dodge Dakota Crew Cab 2010',
    90: 'Dodge Dakota Club Cab 2007',
    91: 'Dodge Magnum Wagon 2008',
    92: 'Dodge Challenger SRT8 2011',
    93: 'Dodge Durango SUV 2012',
    94: 'Dodge Durango SUV 2007',
    95: 'Dodge Charger Sedan 2012',
    96: 'Dodge Charger SRT-8 2009',
    97: 'Eagle Talon Hatchback 1998',
    98: 'FIAT 500 Abarth 2012',
    99: 'FIAT 500 Convertible 2012',
    100: 'Ferrari FF Coupe 2012',
    101: 'Ferrari California Convertible 2012',
    102: 'Ferrari 458 Italia Convertible 2012',
    103: 'Ferrari 458 Italia Coupe 2012',
    104: 'Fisker Karma Sedan 2012',
    105: 'Ford F-450 Super Duty Crew Cab 2012',
    106: 'Ford Mustang Convertible 2007',
    107: 'Ford Freestar Minivan 2007',
    108: 'Ford Expedition EL SUV 2009',
    109: 'Ford Edge SUV 2012',
    110: 'Ford Ranger SuperCab 2011',
    111: 'Ford GT Coupe 2006',
    112: 'Ford F-150 Regular Cab 2012',
    113: 'Ford F-150 Regular Cab 2007',
    114: 'Ford Focus Sedan 2007',
    115: 'Ford E-Series Wagon Van 2012',
    116: 'Ford Fiesta Sedan 2012',
    117: 'GMC Terrain SUV 2012',
    118: 'GMC Savana Van 2012',
    119: 'GMC Yukon Hybrid SUV 2012',
    120: 'GMC Acadia SUV 2012',
    121: 'GMC Canyon Extended Cab 2012',
    122: 'Geo Metro Convertible 1993',
    123: 'HUMMER H3T Crew Cab 2010',
    124: 'HUMMER H2 SUT Crew Cab 2009',
    125: 'Honda Odyssey Minivan 2012',
    126: 'Honda Odyssey Minivan 2007',
    127: 'Honda Accord Coupe 2012',
    128: 'Honda Accord Sedan 2012',
    129: 'Hyundai Veloster Hatchback 2012',
    130: 'Hyundai Santa Fe SUV 2012',
    131: 'Hyundai Tucson SUV 2012',
    132: 'Hyundai Veracruz SUV 2012',
    133: 'Hyundai Sonata Hybrid Sedan 2012',
    134: 'Hyundai Elantra Sedan 2007',
    135: 'Hyundai Accent Sedan 2012',
    136: 'Hyundai Genesis Sedan 2012',
    137: 'Hyundai Sonata Sedan 2012',
    138: 'Hyundai Elantra Touring Hatchback 2012',
    139: 'Hyundai Azera Sedan 2012',
    140: 'Infiniti G Coupe IPL 2012',
    141: 'Infiniti QX56 SUV 2011',
    142: 'Isuzu Ascender SUV 2008',
    143: 'Jaguar XK XKR 2012',
    144: 'Jeep Patriot SUV 2012',
    145: 'Jeep Wrangler SUV 2012',
    146: 'Jeep Liberty SUV 2012',
    147: 'Jeep Grand Cherokee SUV 2012',
    148: 'Jeep Compass SUV 2012',
    149: 'Lamborghini Reventon Coupe 2008',
    150: 'Lamborghini Aventador Coupe 2012',
    151: 'Lamborghini Gallardo LP 570-4 Superleggera 2012',
    152: 'Lamborghini Diablo Coupe 2001',
    153: 'Land Rover Range Rover SUV 2012',
    154: 'Land Rover LR2 SUV 2012',
    155: 'Lincoln Town Car Sedan 2011',
    156: 'MINI Cooper Roadster Convertible 2012',
    157: 'Maybach Landaulet Convertible 2012',
    158: 'Mazda Tribute SUV 2011',
    159: 'McLaren MP4-12C Coupe 2012',
    160: 'Mercedes-Benz 300-Class Convertible 1993',
    161: 'Mercedes-Benz C-Class Sedan 2012',
    162: 'Mercedes-Benz SL-Class Coupe 2009',
    163: 'Mercedes-Benz E-Class Sedan 2012',
    164: 'Mercedes-Benz S-Class Sedan 2012',
    165: 'Mercedes-Benz Sprinter Van 2012',
    166: 'Mitsubishi Lancer Sedan 2012',
    167: 'Nissan Leaf Hatchback 2012',
    168: 'Nissan NV Passenger Van 2012',
    169: 'Nissan Juke Hatchback 2012',
    170: 'Nissan 240SX Coupe 1998',
    171: 'Plymouth Neon Coupe 1999',
    172: 'Porsche Panamera Sedan 2012',
    173: 'Ram C/V Cargo Van Minivan 2012',
    174: 'Rolls-Royce Phantom Drophead Coupe Convertible 2012',
    175: 'Rolls-Royce Ghost Sedan 2012',
    176: 'Rolls-Royce Phantom Sedan 2012',
    177: 'Scion xD Hatchback 2012',
    178: 'Spyker C8 Convertible 2009',
    179: 'Spyker C8 Coupe 2009',
    180: 'Suzuki Aerio Sedan 2007',
    181: 'Suzuki Kizashi Sedan 2012',
    182: 'Suzuki SX4 Hatchback 2012',
    183: 'Suzuki SX4 Sedan 2012',
    184: 'Tesla Model S Sedan 2012',
    185: 'Toyota Sequoia SUV 2012',
    186: 'Toyota Camry Sedan 2012',
    187: 'Toyota Corolla Sedan 2012',
    188: 'Toyota 4Runner SUV 2012',
    189: 'Volkswagen Golf Hatchback 2012',
    190: 'Volkswagen Golf Hatchback 1991',
    191: 'Volkswagen Beetle Hatchback 2012',
    192: 'Volvo C30 Hatchback 2012',
    193: 'Volvo 240 Sedan 1993',
    194: 'Volvo XC90 SUV 2007',
    195: 'smart fortwo Convertible 2012'
}

cars_templates = [
    'a photo of a {}.',
    'a photo of the {}.',
    'a photo of my {}.',
    'i love my {}!',
    'a photo of my dirty {}.',
    'a photo of my clean {}.',
    'a photo of my new {}.',
    'a photo of my old {}.',
]

cifar10_classes_map = {
    0:'airplane',
    1:'automobile',
    2:'bird',
    3:'cat',
    4:'deer',
    5:'dog',
    6:'frog',
    7:'horse',
    8:'ship',
    9:'truck',
}

cifar10_templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]

gtsrb_classes_map = {
    0:'red and white circle 20 kph speed limit',
    1:'red and white circle 30 kph speed limit',
    2:'red and white circle 50 kph speed limit',
    3:'red and white circle 60 kph speed limit',
    4:'red and white circle 70 kph speed limit',
    5:'red and white circle 80 kph speed limit',
    6:'end / de-restriction of 80 kph speed limit',
    7:'red and white circle 100 kph speed limit',
    8:'red and white circle 120 kph speed limit',
    9:'red and white circle red car and black car no passing',
    10:'red and white circle red truck and black car no passing',
    11:'red and white triangle road intersection warning',
    12:'white and yellow diamond priority road',
    13:'red and white upside down triangle yield right-of-way',
    14:'stop',
    15:'empty red and white circle',
    16:'red and white circle no truck entry',
    17:'red circle with white horizonal stripe no entry',
    18:'red and white triangle with exclamation mark warning',
    19:'red and white triangle with black left curve approaching warning',
    20:'red and white triangle with black right curve approaching warning',
    21:'red and white triangle with black double curve approaching warning',
    22:'red and white triangle rough / bumpy road warning',
    23:'red and white triangle car skidding / slipping warning',
    24:'red and white triangle with merging / narrow lanes warning',
    25:'red and white triangle with person digging / construction / road work warning',
    26:'red and white triangle with traffic light approaching warning',
    27:'red and white triangle with person walking warning',
    28:'red and white triangle with child and person walking warning',
    29:'red and white triangle with bicyle warning',
    30:'red and white triangle with snowflake / ice warning',
    31:'red and white triangle with deer warning',
    32:'white circle with gray strike bar no speed limit',
    33:'blue circle with white right turn arrow mandatory',
    34:'blue circle with white left turn arrow mandatory',
    35:'blue circle with white forward arrow mandatory',
    36:'blue circle with white forward or right turn arrow mandatory',
    37:'blue circle with white forward or left turn arrow mandatory',
    38:'blue circle with white keep right arrow mandatory',
    39:'blue circle with white keep left arrow mandatory',
    40:'blue circle with white arrows indicating a traffic circle',
    41:'white circle with gray strike bar indicating no passing for cars has ended',
    42:'white circle with gray strike bar indicating no passing for trucks has ended',
}

gtsrb_templates = [
    'a zoomed in photo of a "{}" traffic sign.',
    'a centered photo of a "{}" traffic sign.',
    'a close up photo of a "{}" traffic sign.',
]

cifar100_class_map = {
    0: "apple", 1: "aquarium fish", 2: "baby", 3: "bear", 4: "beaver",
    5: "bed", 6: "bee", 7: "beetle", 8: "bicycle", 9: "bottle",
    10: "bowl", 11: "boy", 12: "bridge", 13: "bus", 14: "butterfly",
    15: "camel", 16: "can", 17: "castle", 18: "caterpillar", 19: "cattle",
    20: "chair", 21: "chimpanzee", 22: "clock", 23: "cloud", 24: "cockroach",
    25: "couch", 26: "crab", 27: "crocodile", 28: "cup", 29: "dinosaur",
    30: "dolphin", 31: "elephant", 32: "flatfish", 33: "forest", 34: "fox",
    35: "girl", 36: "hamster", 37: "house", 38: "kangaroo", 39: "keyboard",
    40: "lamp", 41: "lawn mower", 42: "leopard", 43: "lion", 44: "lizard",
    45: "lobster", 46: "man", 47: "maple tree", 48: "motorcycle", 49: "mountain",
    50: "mouse", 51: "mushroom", 52: "oak tree", 53: "orange", 54: "orchid",
    55: "otter", 56: "palm tree", 57: "pear", 58: "pickup truck", 59: "pine tree",
    60: "plain", 61: "plate", 62: "poppy", 63: "porcupine", 64: "possum",
    65: "rabbit", 66: "raccoon", 67: "ray", 68: "road", 69: "rocket",
    70: "rose", 71: "sea", 72: "seal", 73: "shark", 74: "shrew",
    75: "skunk", 76: "skyscraper", 77: "snail", 78: "snake", 79: "spider",
    80: "squirrel", 81: "streetcar", 82: "sunflower", 83: "sweet pepper", 84: "table",
    85: "tank", 86: "telephone", 87: "television", 88: "tiger", 89: "tractor",
    90: "train", 91: "trout", 92: "tulip", 93: "turtle", 94: "wardrobe",
    95: "whale", 96: "willow tree", 97: "wolf", 98: "woman", 99: "worm"
}

cifar100_templates = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'a low contrast photo of a {c}.',
    lambda c: f'a high contrast photo of a {c}.',
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a photo of a big {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a low contrast photo of the {c}.',
    lambda c: f'a high contrast photo of the {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the big {c}.',
]

flower102_classes_map = {
    0: 'pink primrose', 1: 'hard-leaved pocket orchid', 2: 'canterbury bells', 
    3: 'sweet pea', 4: 'english marigold', 5: 'tiger lily', 6: 'moon orchid', 
    7: 'bird of paradise', 8: 'monkshood', 9: 'globe thistle', 10: 'snapdragon', 
    11: "colt's foot", 12: 'king protea', 13: 'spear thistle', 14: 'yellow iris', 
    15: 'globe flower', 16: 'purple coneflower', 17: 'peruvian lily', 
    18: 'balloon flower', 19: 'giant white arum lily', 20: 'fire lily', 
    21: 'pincushion flower', 22: 'fritillary', 23: 'red ginger', 24: 'grape hyacinth', 
    25: 'corn poppy', 26: 'prince of wales feathers', 27: 'stemless gentian', 
    28: 'artichoke', 29: 'sweet william', 30: 'carnation', 31: 'garden phlox', 
    32: 'love in the mist', 33: 'mexican aster', 34: 'alpine sea holly', 
    35: 'ruby-lipped cattleya', 36: 'cape flower', 37: 'great masterwort', 
    38: 'siam tulip', 39: 'lenten rose', 40: 'barbeton daisy', 41: 'daffodil', 
    42: 'sword lily', 43: 'poinsettia', 44: 'bolero deep blue', 45: 'wallflower', 
    46: 'marigold', 47: 'buttercup', 48: 'oxeye daisy', 49: 'common dandelion', 
    50: 'petunia', 51: 'wild pansy', 52: 'primula', 53: 'sunflower', 
    54: 'pelargonium', 55: 'bishop of llandaff', 56: 'gaura', 57: 'geranium', 
    58: 'orange dahlia', 59: 'pink and yellow dahlia', 60: 'cautleya spicata', 
    61: 'japanese anemone', 62: 'black-eyed susan', 63: 'silverbush', 
    64: 'californian poppy', 65: 'osteospermum', 66: 'spring crocus', 
    67: 'bearded iris', 68: 'windflower', 69: 'tree poppy', 70: 'gazania', 
    71: 'azalea', 72: 'water lily', 73: 'rose', 74: 'thorn apple', 
    75: 'morning glory', 76: 'passion flower', 77: 'lotus', 78: 'toad lily', 
    79: 'anthurium', 80: 'frangipani', 81: 'clematis', 82: 'hibiscus', 
    83: 'columbine', 84: 'desert-rose', 85: 'tree mallow', 86: 'magnolia', 
    87: 'cyclamen', 88: 'watercress', 89: 'canna lily', 90: 'hippeastrum', 
    91: 'bee balm', 92: 'air plant', 93: 'foxglove', 94: 'bougainvillea', 
    95: 'camellia', 96: 'mallow', 97: 'mexican petunia', 98: 'bromelia', 
    99: 'blanket flower', 100: 'trumpet creeper', 101: 'blackberry lily'
}

flower102_templates = [
    'a photo of a {}, a type of flower.',
]

food101_classes_map = {
    0: "Apple pie", 1: "Baby back ribs", 2: "Baklava", 3: "Beef carpaccio", 4: "Beef tartare",
    5: "Beet salad", 6: "Beignets", 7: "Bibimbap", 8: "Bread pudding", 9: "Breakfast burrito",
    10: "Bruschetta", 11: "Caesar salad", 12: "Cannoli", 13: "Caprese salad", 14: "Carrot cake",
    15: "Ceviche", 16: "Cheesecake", 17: "Cheese plate", 18: "Chicken curry",
    19: "Chicken quesadilla", 20: "Chicken wings", 21: "Chocolate cake", 22: "Chocolate mousse",
    23: "Churros", 24: "Clam chowder", 25: "Club sandwich", 26: "Crab cakes", 27: "Creme brulee",
    28: "Croque madame", 29: "Cup cakes", 30: "Deviled eggs", 31: "Donuts", 32: "Dumplings",
    33: "Edamame", 34: "Eggs benedict", 35: "Escargots", 36: "Falafel", 37: "Filet mignon",
    38: "Fish and chips", 39: "Foie gras", 40: "French fries", 41: "French onion soup",
    42: "French toast", 43: "Fried calamari", 44: "Fried rice", 45: "Frozen yogurt",
    46: "Garlic bread", 47: "Gnocchi", 48: "Greek salad", 49: "Grilled cheese sandwich",
    50: "Grilled salmon", 51: "Guacamole", 52: "Gyoza", 53: "Hamburger", 54: "Hot and sour soup",
    55: "Hot dog", 56: "Huevos rancheros", 57: "Hummus", 58: "Ice cream", 59: "Lasagna",
    60: "Lobster bisque", 61: "Lobster roll sandwich", 62: "Macaroni and cheese", 63: "Macarons",
    64: "Miso soup", 65: "Mussels", 66: "Nachos", 67: "Omelette", 68: "Onion rings",
    69: "Oysters", 70: "Pad thai", 71: "Paella", 72: "Pancakes", 73: "Panna cotta",
    74: "Peking duck", 75: "Pho", 76: "Pizza", 77: "Pork chop", 78: "Poutine", 79: "Prime rib",
    80: "Pulled pork sandwich", 81: "Ramen", 82: "Ravioli", 83: "Red velvet cake", 84: "Risotto",
    85: "Samosa", 86: "Sashimi", 87: "Scallops", 88: "Seaweed salad", 89: "Shrimp and grits",
    90: "Spaghetti bolognese", 91: "Spaghetti carbonara", 92: "Spring rolls", 93: "Steak",
    94: "Strawberry shortcake", 95: "Sushi", 96: "Tacos", 97: "Takoyaki", 98: "Tiramisu",
    99: "Tuna tartare", 100: "Waffles"
}

food101_templates = [
    'a photo of {}, a type of food.',
]


eurosat_classes_map = {            
    0: "annual crop land",
    1: "forest",
    2: "brushland or shrubland",
    3: "highway or road",
    4: "industrial buildings or commercial buildings",
    5: "pasture land",
    6: "permanent crop land",
    7: "residential buildings or homes or apartments",
    8: "river",
    9: "lake or sea"
}

eurosat_templates = [
    'a centered satellite photo of {}.',
    'a centered satellite photo of a {}.',
    'a centered satellite photo of the {}.',
]

sun397_classes_map={
    0: "abbey", 1: "airplane cabin", 2: "airport terminal", 3: "alley", 4: "amphitheater",
    5: "amusement arcade", 6: "amusement park", 7: "anechoic chamber", 8: "apartment building outdoor", 9: "apse indoor",
    10: "aquarium", 11: "aqueduct", 12: "arch", 13: "archive", 14: "arrival gate outdoor",
    15: "art gallery", 16: "art school", 17: "art studio", 18: "assembly line", 19: "athletic field outdoor",
    20: "atrium public", 21: "attic", 22: "auditorium", 23: "auto factory", 24: "badlands",
    25: "badminton court indoor", 26: "baggage claim", 27: "bakery shop", 28: "balcony exterior", 29: "balcony interior",
    30: "ball pit", 31: "ballroom", 32: "bamboo forest", 33: "banquet hall", 34: "bar",
    35: "barn", 36: "barndoor", 37: "baseball field", 38: "basement", 39: "basilica",
    40: "basketball court outdoor", 41: "bathroom", 42: "batters box", 43: "bayou", 44: "bazaar indoor",
    45: "bazaar outdoor", 46: "beach", 47: "beauty salon", 48: "bedroom", 49: "berth",
    50: "biology laboratory", 51: "bistro indoor", 52: "boardwalk", 53: "boat deck", 54: "boathouse",
    55: "bookstore", 56: "booth indoor", 57: "botanical garden", 58: "bow window indoor", 59: "bow window outdoor",
    60: "bowling alley", 61: "boxing ring", 62: "brewery indoor", 63: "bridge", 64: "building facade",
    65: "bullring", 66: "burial chamber", 67: "bus interior", 68: "butchers shop", 69: "butte",
    70: "cabin outdoor", 71: "cafeteria", 72: "campsite", 73: "campus", 74: "canal natural",
    75: "canal urban", 76: "candy store", 77: "canyon", 78: "car interior backseat", 79: "car interior frontseat",
    80: "carrousel", 81: "casino indoor", 82: "castle", 83: "catacomb", 84: "cathedral indoor",
    85: "cathedral outdoor", 86: "cavern indoor", 87: "cemetery", 88: "chalet", 89: "cheese factory",
    90: "chemistry lab", 91: "chicken coop indoor", 92: "chicken coop outdoor", 93: "childs room", 94: "church indoor",
    95: "church outdoor", 96: "classroom", 97: "clean room", 98: "cliff", 99: "cloister indoor",
    100: "closet", 101: "clothing store", 102: "coast", 103: "cockpit", 104: "coffee shop",
    105: "computer room", 106: "conference center", 107: "conference room", 108: "construction site", 109: "control room",
    110: "control tower outdoor", 111: "corn field", 112: "corral", 113: "corridor", 114: "cottage garden",
    115: "courthouse", 116: "courtroom", 117: "courtyard", 118: "covered bridge exterior", 119: "creek",
    120: "crevasse", 121: "crosswalk", 122: "cubicle office", 123: "dam", 124: "delicatessen",
    125: "dentists office", 126: "desert sand", 127: "desert vegetation", 128: "diner indoor", 129: "diner outdoor",
    130: "dinette home", 131: "dinette vehicle", 132: "dining car", 133: "dining room", 134: "discotheque",
    135: "dock", 136: "doorway outdoor", 137: "dorm room", 138: "driveway", 139: "driving range outdoor",
    140: "drugstore", 141: "electrical substation", 142: "elevator door", 143: "elevator interior", 144: "elevator shaft",
    145: "engine room", 146: "escalator indoor", 147: "excavation", 148: "factory indoor", 149: "fairway",
    150: "fastfood restaurant", 151: "field cultivated", 152: "field wild", 153: "fire escape", 154: "fire station",
    155: "firing range indoor", 156: "fishpond", 157: "florist shop indoor", 158: "food court", 159: "forest broadleaf",
    160: "forest needleleaf", 161: "forest path", 162: "forest road", 163: "formal garden", 164: "fountain",
    165: "galley", 166: "game room", 167: "garage indoor", 168: "garbage dump", 169: "gas station",
    170: "gazebo exterior", 171: "general store indoor", 172: "general store outdoor", 173: "gift shop", 174: "golf course",
    175: "greenhouse indoor", 176: "greenhouse outdoor", 177: "gymnasium indoor", 178: "hangar indoor", 179: "hangar outdoor",
    180: "harbor", 181: "hayfield", 182: "heliport", 183: "herb garden", 184: "highway",
    185: "hill", 186: "home office", 187: "hospital", 188: "hospital room", 189: "hot spring",
    190: "hot tub outdoor", 191: "hotel outdoor", 192: "hotel room", 193: "house", 194: "hunting lodge outdoor",
    195: "ice cream parlor", 196: "ice floe", 197: "ice shelf", 198: "ice skating rink indoor", 199: "ice skating rink outdoor",
    200: "iceberg", 201: "igloo", 202: "industrial area", 203: "inn outdoor", 204: "islet",
    205: "jacuzzi indoor", 206: "jail cell", 207: "jail indoor", 208: "jewelry shop", 209: "kasbah",
    210: "kennel indoor", 211: "kennel outdoor", 212: "kindergarden classroom", 213: "kitchen", 214: "kitchenette",
    215: "labyrinth outdoor", 216: "lake natural", 217: "landfill", 218: "landing deck", 219: "laundromat",
    220: "lecture room", 221: "library indoor", 222: "library outdoor", 223: "lido deck outdoor", 224: "lift bridge",
    225: "lighthouse", 226: "limousine interior", 227: "living room", 228: "lobby", 229: "lock chamber",
    230: "locker room", 231: "mansion", 232: "manufactured home", 233: "market indoor", 234: "market outdoor",
    235: "marsh", 236: "martial arts gym", 237: "mausoleum", 238: "medina", 239: "moat water",
    240: "monastery outdoor", 241: "mosque indoor", 242: "mosque outdoor", 243: "motel", 244: "mountain",
    245: "mountain snowy", 246: "movie theater indoor", 247: "museum indoor", 248: "music store", 249: "music studio",
    250: "nuclear power plant outdoor", 251: "nursery", 252: "oast house", 253: "observatory outdoor", 254: "ocean",
    255: "office", 256: "office building", 257: "oil refinery outdoor", 258: "oilrig", 259: "operating room",
    260: "orchard", 261: "outhouse outdoor", 262: "pagoda", 263: "palace", 264: "pantry",
    265: "park", 266: "parking garage indoor", 267: "parking garage outdoor", 268: "parking lot", 269: "parlor",
    270: "pasture", 271: "patio", 272: "pavilion", 273: "pharmacy", 274: "phone booth",
    275: "physics laboratory", 276: "picnic area", 277: "pilothouse indoor", 278: "planetarium outdoor", 279: "playground",
    280: "playroom", 281: "plaza", 282: "podium indoor", 283: "podium outdoor", 284: "pond",
    285: "poolroom establishment", 286: "poolroom home", 287: "power plant outdoor", 288: "promenade deck", 289: "pub indoor",
    290: "pulpit", 291: "putting green", 292: "racecourse", 293: "raceway", 294: "raft",
    295: "railroad track", 296: "rainforest", 297: "reception", 298: "recreation room", 299: "residential neighborhood",
    300: "restaurant", 301: "restaurant kitchen", 302: "restaurant patio", 303: "rice paddy", 304: "riding arena",
    305: "river", 306: "rock arch", 307: "rope bridge", 308: "ruin", 309: "runway",
    310: "sandbar", 311: "sandbox", 312: "sauna", 313: "schoolhouse", 314: "sea cliff",
    315: "server room", 316: "shed", 317: "shoe shop", 318: "shopfront", 319: "shopping mall indoor",
    320: "shower", 321: "skatepark", 322: "ski lodge", 323: "ski resort", 324: "ski slope",
    325: "sky", 326: "skyscraper", 327: "slum", 328: "snowfield", 329: "squash court",
    330: "stable", 331: "stadium baseball", 332: "stadium football", 333: "stage indoor", 334: "staircase",
    335: "street", 336: "subway interior", 337: "subway station platform", 338: "supermarket", 339: "sushi bar",
    340: "swamp", 341: "swimming pool indoor", 342: "swimming pool outdoor", 343: "synagogue indoor", 344: "synagogue outdoor",
    345: "television studio", 346: "temple east asia", 347: "temple south asia", 348: "tennis court indoor", 349: "tennis court outdoor",
    350: "tent outdoor", 351: "theater indoor procenium", 352: "theater indoor seats", 353: "thriftshop", 354: "throne room",
    355: "ticket booth", 356: "toll plaza", 357: "topiary garden", 358: "tower", 359: "toyshop",
    360: "track outdoor", 361: "train railway", 362: "train station platform", 363: "tree farm", 364: "tree house",
    365: "trench", 366: "underwater coral reef", 367: "utility room", 368: "valley", 369: "van interior",
    370: "vegetable garden", 371: "veranda", 372: "veterinarians office", 373: "viaduct", 374: "videostore",
    375: "village", 376: "vineyard", 377: "volcano", 378: "volleyball court indoor", 379: "volleyball court outdoor",
    380: "waiting room", 381: "warehouse indoor", 382: "water tower", 383: "waterfall block", 384: "waterfall fan",
    385: "waterfall plunge", 386: "watering hole", 387: "wave", 388: "wet bar", 389: "wheat field",
    390: "wind farm", 391: "windmill", 392: "wine cellar barrel storage", 393: "wine cellar bottle storage", 394: "wrestling ring indoor",
    395: "yard", 396: "youth hostel"
}

sun397_templates = [
    'a photo of a {}.',
    'a photo of the {}.',
]

oxford_pets_classes_map = {
    0: 'Abyssinian', 1: 'Bengal', 2: 'Birman', 3: 'Bombay', 4: 'British Shorthair',
    5: 'Egyptian Mau', 6: 'Maine Coon', 7: 'Persian', 8: 'Ragdoll', 9: 'Russian_Blue',
    10: 'Siamese', 11: 'Sphynx', 12: 'american bulldog', 13: 'american pit bull terrier', 14: 'basset hound',
    15: 'beagle', 16: 'boxer', 17: 'chihuahua', 18: 'english cocker spaniel', 19: 'english setter',
    20: 'german shorthaired', 21: 'great pyrenees', 22: 'havanese', 23: 'japanese chin', 24: 'keeshond',
    25: 'leonberger', 26: 'miniature pinscher', 27: 'newfoundland', 28: 'pomeranian', 29: 'pug',
    30: 'saint bernard', 31: 'samoyed', 32: 'scottish terrier', 33: 'shiba inu', 34: 'staffordshire bull terrier',
    35: 'wheaten terrier', 36: 'yorkshire terrier'
}

oxford_pets_templates = [
    'a photo of a {}, a type of pet.',
]

resisc45_classes_map = {
    0: "airplane", 1: "airport", 2: "baseball diamond", 3: "basketball court", 4: "beach",
    5: "bridge", 6: "chaparral", 7: "church", 8: "circular farmland", 9: "cloud",
    10: "commercial area", 11: "dense residential", 12: "desert", 13: "forest", 14: "freeway",
    15: "golf course", 16: "ground track field", 17: "harbor", 18: "industrial area", 19: "intersection",
    20: "island", 21: "lake", 22: "meadow", 23: "medium residential", 24: "mobile home park",
    25: "mountain", 26: "overpass", 27: "palace", 28: "parking lot", 29: "railway",
    30: "railway station", 31: "rectangular farmland", 32: "river", 33: "roundabout", 34: "runway",
    35: "sea ice", 36: "ship", 37: "snowberg", 38: "sparse residential", 39: "stadium",
    40: "storage tank", 41: "tennis court", 42: "terrace", 43: "thermal power station", 44: "wetland"
}

resisc45_templates = [
    'satellite imagery of {}.',
    'aerial imagery of {}.',
    'satellite photo of {}.',
    'aerial photo of {}.',
    'satellite view of {}.',
    'aerial view of {}.',
    'satellite imagery of a {}.',
    'aerial imagery of a {}.',
    'satellite photo of a {}.',
    'aerial photo of a {}.',
    'satellite view of a {}.',
    'aerial view of a {}.',
    'satellite imagery of the {}.',
    'aerial imagery of the {}.',
    'satellite photo of the {}.',
    'aerial photo of the {}.',
    'satellite view of the {}.',
    'aerial view of the {}.',
]

country211_classes_map = {
    0: "Andorra", 1: "United Arab Emirates", 2: "Afghanistan", 3: "Antigua and Barbuda", 4: "Anguilla",
    5: "Albania", 6: "Armenia", 7: "Angola", 8: "Antarctica", 9: "Argentina",
    10: "Austria", 11: "Australia", 12: "Aruba", 13: "Aland Islands", 14: "Azerbaijan",
    15: "Bosnia and Herzegovina", 16: "Barbados", 17: "Bangladesh", 18: "Belgium", 19: "Burkina Faso",
    20: "Bulgaria", 21: "Bahrain", 22: "Benin", 23: "Bermuda", 24: "Brunei Darussalam",
    25: "Bolivia", 26: "Bonaire, Saint Eustatius and Saba", 27: "Brazil", 28: "Bahamas", 29: "Bhutan",
    30: "Botswana", 31: "Belarus", 32: "Belize", 33: "Canada", 34: "DR Congo",
    35: "Central African Republic", 36: "Switzerland", 37: "Cote d'Ivoire", 38: "Cook Islands", 39: "Chile",
    40: "Cameroon", 41: "China", 42: "Colombia", 43: "Costa Rica", 44: "Cuba",
    45: "Cabo Verde", 46: "Curacao", 47: "Cyprus", 48: "Czech Republic", 49: "Germany",
    50: "Denmark", 51: "Dominica", 52: "Dominican Republic", 53: "Algeria", 54: "Ecuador",
    55: "Estonia", 56: "Egypt", 57: "Spain", 58: "Ethiopia", 59: "Finland",
    60: "Fiji", 61: "Falkland Islands", 62: "Faeroe Islands", 63: "France", 64: "Gabon",
    65: "United Kingdom", 66: "Grenada", 67: "Georgia", 68: "French Guiana", 69: "Guernsey",
    70: "Ghana", 71: "Gibraltar", 72: "Greenland", 73: "Gambia", 74: "Guadeloupe",
    75: "Greece", 76: "South Georgia and South Sandwich Is.", 77: "Guatemala", 78: "Guam", 79: "Guyana",
    80: "Hong Kong", 81: "Honduras", 82: "Croatia", 83: "Haiti", 84: "Hungary",
    85: "Indonesia", 86: "Ireland", 87: "Israel", 88: "Isle of Man", 89: "India",
    90: "Iraq", 91: "Iran", 92: "Iceland", 93: "Italy", 94: "Jersey",
    95: "Jamaica", 96: "Jordan", 97: "Japan", 98: "Kenya", 99: "Kyrgyz Republic",
    100: "Cambodia", 101: "St. Kitts and Nevis", 102: "North Korea", 103: "South Korea", 104: "Kuwait",
    105: "Cayman Islands", 106: "Kazakhstan", 107: "Laos", 108: "Lebanon", 109: "St. Lucia",
    110: "Liechtenstein", 111: "Sri Lanka", 112: "Liberia", 113: "Lithuania", 114: "Luxembourg",
    115: "Latvia", 116: "Libya", 117: "Morocco", 118: "Monaco", 119: "Moldova",
    120: "Montenegro", 121: "Saint-Martin", 122: "Madagascar", 123: "Macedonia", 124: "Mali",
    125: "Myanmar", 126: "Mongolia", 127: "Macau", 128: "Martinique", 129: "Mauritania",
    130: "Malta", 131: "Mauritius", 132: "Maldives", 133: "Malawi", 134: "Mexico",
    135: "Malaysia", 136: "Mozambique", 137: "Namibia", 138: "New Caledonia", 139: "Nigeria",
    140: "Nicaragua", 141: "Netherlands", 142: "Norway", 143: "Nepal", 144: "New Zealand",
    145: "Oman", 146: "Panama", 147: "Peru", 148: "French Polynesia", 149: "Papua New Guinea",
    150: "Philippines", 151: "Pakistan", 152: "Poland", 153: "Puerto Rico", 154: "Palestine",
    155: "Portugal", 156: "Palau", 157: "Paraguay", 158: "Qatar", 159: "Reunion",
    160: "Romania", 161: "Serbia", 162: "Russia", 163: "Rwanda", 164: "Saudi Arabia",
    165: "Solomon Islands", 166: "Seychelles", 167: "Sudan", 168: "Sweden", 169: "Singapore",
    170: "St. Helena", 171: "Slovenia", 172: "Svalbard and Jan Mayen Islands", 173: "Slovakia", 174: "Sierra Leone",
    175: "San Marino", 176: "Senegal", 177: "Somalia", 178: "South Sudan", 179: "El Salvador",
    180: "Sint Maarten", 181: "Syria", 182: "Eswatini", 183: "Togo", 184: "Thailand",
    185: "Tajikistan", 186: "Timor-Leste", 187: "Turkmenistan", 188: "Tunisia", 189: "Tonga",
    190: "Turkey", 191: "Trinidad and Tobago", 192: "Taiwan", 193: "Tanzania", 194: "Ukraine",
    195: "Uganda", 196: "United States", 197: "Uruguay", 198: "Uzbekistan", 199: "Vatican",
    200: "Venezuela", 201: "British Virgin Islands", 202: "United States Virgin Islands", 203: "Vietnam", 204: "Vanuatu",
    205: "Samoa", 206: "Kosovo", 207: "Yemen", 208: "South Africa", 209: "Zambia",
    210: "Zimbabwe"
}

country211_templates = [
    'a photo i took in {}.',
    'a photo i took while visiting {}.',
    'a photo from my home country of {}.',
    'a photo from my visit to {}.',
    'a photo showing the country of {}.',
]



#################
def cifar100_calibration_collate_fn(batch):
    images = torch.stack([example['img']['pixel_values'] for example in batch])
    # template=random.choice(cifar100_templates)
    template=cifar100_templates[0]
    texts = [template.format(cifar100_class_map[example['fine_label']]) if isinstance(template, str) else template(
            cifar100_class_map[example['fine_label']]) for example in batch]
    return {"pixel_values": images, "texts": texts}

def cifar100_collate_fn(batch):
    images = torch.stack([example['img']['pixel_values'] for example in batch])
    labels = torch.tensor([example['fine_label'] for example in batch])
    return {"pixel_values": images, "labels": labels}

def cifar100_transforms(examples):
    examples["img"] = [processor(images=image, return_tensors="pt") for image in examples["img"]]
    return examples

def cifar10_calibration_collate_fn(batch):
    images = torch.stack([example['img']['pixel_values'] for example in batch])
    # template=random.choice(cifar100_templates)
    template=cifar10_templates[0]
    texts = [template.format(cifar10_classes_map[example['label']]) if isinstance(template, str) else template(
            cifar10_classes_map[example['label']]) for example in batch]
    return {"pixel_values": images, "texts": texts}

def cifar10_collate_fn(batch):
    images = torch.stack([example['img']['pixel_values'] for example in batch])
    labels = torch.tensor([example['label'] for example in batch])
    return {"pixel_values": images, "labels": labels}

def cifar10_transforms(examples):
    examples["img"] = [processor(images=image, return_tensors="pt") for image in examples["img"]]
    return examples


def flower102_calibration_collate_fn(batch):
    images = torch.stack([example['image']['pixel_values'] for example in batch])
    template=flower102_templates[0]
    # template=random.choice(flower102_templates)
    texts = [template.format(flower102_classes_map[example['label']]) if isinstance(template, str) else template(
            flower102_classes_map[example['label']]) for example in batch]
    return {"pixel_values": images, "texts": texts}

def food101_calibration_collate_fn(batch):
    images = torch.stack([example['image']['pixel_values'] for example in batch])
    template=food101_templates[0]
    # template=random.choice(food101_templates)
    texts = [template.format(food101_classes_map[example['label']]) if isinstance(template, str) else template(
            food101_classes_map[example['label']]) for example in batch]
    return {"pixel_values": images, "texts": texts}

def eurosat_calibration_collate_fn(batch):
    images = torch.stack([example['image']['pixel_values'] for example in batch])
    template=eurosat_templates[0]
    # template=random.choice(eurosat_templates)
    texts = [template.format(eurosat_classes_map[example['label']]) if isinstance(template, str) else template(
            eurosat_classes_map[example['label']]) for example in batch]
    return {"pixel_values": images, "texts": texts}

def sun397_calibration_collate_fn(batch):
    images = torch.stack([example['image']['pixel_values'] for example in batch])
    template=sun397_templates[0]
    # template=random.choice(sun397_templates)
    texts = [template.format(sun397_classes_map[example['label']]) if isinstance(template, str) else template(
            sun397_classes_map[example['label']]) for example in batch]
    return {"pixel_values": images, "texts": texts}

def oxford_pets_calibration_collate_fn(batch):
    images = torch.stack([example['image']['pixel_values'] for example in batch])
    template=oxford_pets_templates[0]
    # template=random.choice(oxford_pets_templates)
    texts = [template.format(oxford_pets_classes_map[example['label']]) if isinstance(template, str) else template(
            oxford_pets_classes_map[example['label']]) for example in batch]
    return {"pixel_values": images, "texts": texts}

def cars_calibration_collate_fn(batch):
    images = torch.stack([example['image']['pixel_values'] for example in batch])
    template=cars_templates[0]
    texts = [template.format(cars_classes_map[example['label']]) if isinstance(template, str) else template(
            cars_classes_map[example['label']]) for example in batch]
    return {"pixel_values": images, "texts": texts}

def svhn_calibration_collate_fn(batch):
    images = torch.stack([example['image']['pixel_values'] for example in batch])
    template=svhn_templates[0]
    texts = [template.format(svhn_classes_map[example['label']]) if isinstance(template, str) else template(
            svhn_classes_map[example['label']]) for example in batch]
    return {"pixel_values": images, "texts": texts}

def mnist_calibration_collate_fn(batch):
    images = torch.stack([example['image']['pixel_values'] for example in batch])
    template=mnist_templates[0]
    texts = [template.format(mnist_classes_map[example['label']]) if isinstance(template, str) else template(
            mnist_classes_map[example['label']]) for example in batch]
    return {"pixel_values": images, "texts": texts}

def resisc45_calibration_collate_fn(batch):
    images = torch.stack([example['image']['pixel_values'] for example in batch])
    template=resisc45_templates[0]
    # template=random.choice(resisc45_templates)
    texts = [template.format(resisc45_classes_map[example['label']]) if isinstance(template, str) else template(
            resisc45_classes_map[example['label']]) for example in batch]
    return {"pixel_values": images, "texts": texts}

def country211_calibration_collate_fn(batch):
    images = torch.stack([example['image']['pixel_values'] for example in batch])
    template=country211_templates[0]
    # template=random.choice(country211_templates)
    texts = [template.format(country211_classes_map[example['label']]) if isinstance(template, str) else template(
            country211_classes_map[example['label']]) for example in batch]
    return {"pixel_values": images, "texts": texts}


def gtsrb_calibration_collate_fn(batch):
    images = torch.stack([example['image']['pixel_values'] for example in batch])
    template=gtsrb_templates[0]
    # template=random.choice(country211_templates)
    texts = [template.format(gtsrb_classes_map[example['label']]) if isinstance(template, str) else template(
            gtsrb_classes_map[example['label']]) for example in batch]
    return {"pixel_values": images, "texts": texts}



def normal_collate_fn(batch):
    images = torch.stack([example['image']['pixel_values'] for example in batch])
    labels = torch.tensor([example['label'] for example in batch])
    return {"pixel_values": images, "labels": labels}

def normal_transforms(examples):
    examples["image"] = [processor(images=image, return_tensors="pt") for image in examples["image"]]
    return examples

def country211_transforms(examples):
    examples["image"] = [processor(images=Image.open(io.BytesIO(image['bytes'])), return_tensors="pt") for image in examples["image"]]
    return examples

      
def get_batch_images_dataloader(data_dir,batch_size,split,data_name,calibration=False):
    if data_name=='cifar100':
        dataset = load_dataset(data_dir,split=split)
        dataset = dataset.with_transform(cifar100_transforms)
        if calibration:
            data_loader = DataLoader(dataset, collate_fn=cifar100_calibration_collate_fn, batch_size=batch_size, shuffle=True,num_workers=1)
        else:
            data_loader = DataLoader(dataset, collate_fn=cifar100_collate_fn, batch_size=batch_size, shuffle=True,num_workers=1)
    elif data_name=='cifar10':
        dataset = load_dataset(data_dir,split=split)
        dataset = dataset.with_transform(cifar10_transforms)
        if calibration:
            data_loader = DataLoader(dataset, collate_fn=cifar10_calibration_collate_fn, batch_size=batch_size, shuffle=True,num_workers=1)
        else:
            data_loader = DataLoader(dataset, collate_fn=cifar10_collate_fn, batch_size=batch_size, shuffle=True,num_workers=1)
    else:
        dataset = load_dataset(data_dir,split=split)
        if data_name=='country211':
            dataset = dataset.with_transform(country211_transforms) 
        else:
            dataset = dataset.with_transform(normal_transforms)     
        if calibration:
            match data_name:
                case "flower102":
                    data_loader = DataLoader(dataset, collate_fn=flower102_calibration_collate_fn, batch_size=batch_size, shuffle=True,num_workers=1)
                case "food101":
                    data_loader = DataLoader(dataset, collate_fn=food101_calibration_collate_fn, batch_size=batch_size, shuffle=True,num_workers=1)
                case "eurosat":
                    data_loader = DataLoader(dataset, collate_fn=eurosat_calibration_collate_fn, batch_size=batch_size, shuffle=True,num_workers=1)
                case "sun397":
                    data_loader = DataLoader(dataset, collate_fn=sun397_calibration_collate_fn, batch_size=batch_size, shuffle=True,num_workers=1)
                case "oxford_pets":
                    data_loader = DataLoader(dataset, collate_fn=oxford_pets_calibration_collate_fn, batch_size=batch_size, shuffle=True,num_workers=1)
                case "cars":
                    data_loader = DataLoader(dataset, collate_fn=cars_calibration_collate_fn, batch_size=batch_size, shuffle=True,num_workers=1)
                case "mnist":
                    data_loader = DataLoader(dataset, collate_fn=mnist_calibration_collate_fn, batch_size=batch_size, shuffle=True,num_workers=1)             
                case "svhn":
                    data_loader = DataLoader(dataset, collate_fn=svhn_calibration_collate_fn, batch_size=batch_size, shuffle=True,num_workers=1)                      
                case "resisc45":
                    data_loader = DataLoader(dataset, collate_fn=resisc45_calibration_collate_fn, batch_size=batch_size, shuffle=True,num_workers=1)
                case "country211":
                    data_loader = DataLoader(dataset, collate_fn=country211_calibration_collate_fn, batch_size=batch_size, shuffle=True,num_workers=1)     
                case "gtsrb":
                    data_loader = DataLoader(dataset, collate_fn=gtsrb_calibration_collate_fn, batch_size=batch_size, shuffle=True,num_workers=1)     
                
                case _:
                    raise RuntimeError("Not support dataset name !")
        else:
            data_loader = DataLoader(dataset, collate_fn=normal_collate_fn, batch_size=batch_size, shuffle=True,num_workers=1)
            
            
    return data_loader


def get_hf_loader(split, data_name, batch_size):
    if data_name=='cifar100':
        if split =='calibration':
            data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/cifar100",batch_size,split='train',data_name=data_name,calibration=True)
        else:
            if split=='train':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/cifar100",batch_size,split='train',data_name=data_name)
            else:
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/cifar100",batch_size,split='test',data_name=data_name)
    elif data_name=='cifar10':
        if split =='calibration':
            data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/cifar10",batch_size,split='train',data_name=data_name,calibration=True)
        else:
            if split=='train':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/cifar10",batch_size,split='train',data_name=data_name)
            else:
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/cifar10",batch_size,split='test',data_name=data_name)      
    elif data_name=='gtsrb':
        if split =='calibration':
            data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/gtsrb",batch_size,split='train',data_name=data_name,calibration=True)
        else:
            if split=='train':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/gtsrb",batch_size,split='train',data_name=data_name)
            else:
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/gtsrb",batch_size,split='test',data_name=data_name)                 
    elif data_name=='flower102':
        if split =='calibration':
            data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/flowers102/data",batch_size,split='train',data_name=data_name,calibration=True)
        else:
            if split=='train':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/flowers102/data",batch_size,split='train',data_name=data_name)
            elif split=='test':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/flowers102/data",batch_size,split='test',data_name=data_name)
            elif split=='validation':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/flowers102/data",batch_size,split='validation',data_name=data_name)
    elif data_name=='food101':
        if split =='calibration':
            data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/food101/data",batch_size,split='train',data_name=data_name,calibration=True)
        else:
            if split=='train':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/food101/data",batch_size,split='train',data_name=data_name)
            else:
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/food101/data",batch_size,split='validation',data_name=data_name)
    elif data_name=='eurosat':
        if split =='calibration':
            data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/eurosat/data",batch_size,split='train',data_name=data_name,calibration=True)
        else:
            if split=='train':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/eurosat/data",batch_size,split='train',data_name=data_name)
            elif split=='test':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/eurosat/data",batch_size,split='test',data_name=data_name)
            elif split=='validation':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/eurosat/data",batch_size,split='validation',data_name=data_name)
    elif data_name=='sun397':
        if split =='calibration':
            data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/sun397/data",batch_size,split='train',data_name=data_name,calibration=True)
        else:
            if split=='train':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/sun397/data",batch_size,split='train',data_name=data_name)
            elif split=='test':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/sun397/data",batch_size,split='test',data_name=data_name)
            elif split=='validation':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/sun397/data",batch_size,split='validation',data_name=data_name)
    elif data_name=='oxford_pets':
        if split =='calibration':
            data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/oxford_pets/data",batch_size,split='train',data_name=data_name,calibration=True)
        else:
            if split=='train':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/oxford_pets/data",batch_size,split='train',data_name=data_name)
            else:
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/oxford_pets/data",batch_size,split='test',data_name=data_name)
    elif data_name=='cars':
        if split =='calibration':
            data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/cars/data",batch_size,split='train',data_name=data_name,calibration=True)
        else:
            if split=='train':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/cars/data",batch_size,split='train',data_name=data_name)
            else:
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/cars/data",batch_size,split='test',data_name=data_name)
    elif data_name=='mnist':
        if split =='calibration':
            data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/mnist/data",batch_size,split='train',data_name=data_name,calibration=True)
        else:
            if split=='train':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/mnist/data",batch_size,split='train',data_name=data_name)
            else:
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/mnist/data",batch_size,split='test',data_name=data_name)
    elif data_name=='svhn':
        if split =='calibration':
            data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/svhn/data",batch_size,split='train',data_name=data_name,calibration=True)
        else:
            if split=='train':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/svhn/data",batch_size,split='train',data_name=data_name)
            else:
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/svhn/data",batch_size,split='test',data_name=data_name)    
    elif data_name=='resisc45':
        if split =='calibration':
            data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/resisc45/data",batch_size,split='train',data_name=data_name,calibration=True)
        else:
            if split=='train':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/resisc45/data",batch_size,split='train',data_name=data_name)
            elif split=='test':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/resisc45/data",batch_size,split='test',data_name=data_name)
            elif split=='validation':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/resisc45/data",batch_size,split='validation',data_name=data_name)
    elif data_name=='country211':
        if split =='calibration':
            data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/country211/data",batch_size,split='train',data_name=data_name,calibration=True)
        else:
            if split=='train':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/country211/data",batch_size,split='train',data_name=data_name)
            elif split=='test':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/country211/data",batch_size,split='test',data_name=data_name)
            elif split=='validation':
                data_loader=get_batch_images_dataloader("/home/jiaxinshi/huggingface_dataset/country211/data",batch_size,split='validation',data_name=data_name)
    return data_loader


def get_split_loader(split, data_name, batch_size, args,preprocess=None):
    # Build Dataset Loader
    global processor
    processor = args.processor
    loader = get_hf_loader(split, data_name, batch_size)
    return loader


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids
