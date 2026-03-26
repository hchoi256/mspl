# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class LVISV05Dataset(CocoDataset):
    """LVIS v0.5 dataset for detection."""

    METAINFO = {
        'classes':
        ('acorn', 'aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock',
         'alcohol', 'alligator', 'almond', 'ambulance', 'amplifier', 'anklet',
         'antenna', 'apple', 'apple_juice', 'applesauce', 'apricot', 'apron',
         'aquarium', 'armband', 'armchair', 'armoire', 'armor', 'artichoke',
         'trash_can', 'ashtray', 'asparagus', 'atomizer', 'avocado', 'award',
         'awning', 'ax', 'baby_buggy', 'basketball_backboard', 'backpack',
         'handbag', 'suitcase', 'bagel', 'bagpipe', 'baguet', 'bait', 'ball',
         'ballet_skirt', 'balloon', 'bamboo', 'banana', 'Band_Aid', 'bandage',
         'bandanna', 'banjo', 'banner', 'barbell', 'barge', 'barrel',
         'barrette', 'barrow', 'baseball_base', 'baseball', 'baseball_bat',
         'baseball_cap', 'baseball_glove', 'basket', 'basketball_hoop',
         'basketball', 'bass_horn', 'bat_(animal)', 'bath_mat', 'bath_towel',
         'bathrobe', 'bathtub', 'batter_(food)', 'battery', 'beachball',
         'bead', 'beaker', 'bean_curd', 'beanbag', 'beanie', 'bear', 'bed',
         'bedspread', 'cow', 'beef_(food)', 'beeper', 'beer_bottle',
         'beer_can', 'beetle', 'bell', 'bell_pepper', 'belt', 'belt_buckle',
         'bench', 'beret', 'bib', 'Bible', 'bicycle', 'visor', 'binder',
         'binoculars', 'bird', 'birdfeeder', 'birdbath', 'birdcage',
         'birdhouse', 'birthday_cake', 'birthday_card', 'biscuit_(bread)',
         'pirate_flag', 'black_sheep', 'blackboard', 'blanket', 'blazer',
         'blender', 'blimp', 'blinker', 'blueberry', 'boar', 'gameboard',
         'boat', 'bobbin', 'bobby_pin', 'boiled_egg', 'bolo_tie', 'deadbolt',
         'bolt', 'bonnet', 'book', 'book_bag', 'bookcase', 'booklet',
         'bookmark', 'boom_microphone', 'boot', 'bottle', 'bottle_opener',
         'bouquet', 'bow_(weapon)', 'bow_(decorative_ribbons)', 'bow-tie',
         'bowl', 'pipe_bowl', 'bowler_hat', 'bowling_ball', 'bowling_pin',
         'boxing_glove', 'suspenders', 'bracelet', 'brass_plaque', 'brassiere',
         'bread-bin', 'breechcloth', 'bridal_gown', 'briefcase',
         'bristle_brush', 'broccoli', 'broach', 'broom', 'brownie',
         'brussels_sprouts', 'bubble_gum', 'bucket', 'horse_buggy', 'bull',
         'bulldog', 'bulldozer', 'bullet_train', 'bulletin_board',
         'bulletproof_vest', 'bullhorn', 'corned_beef', 'bun', 'bunk_bed',
         'buoy', 'burrito', 'bus_(vehicle)', 'business_card', 'butcher_knife',
         'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabana', 'cabin_car',
         'cabinet', 'locker', 'cake', 'calculator', 'calendar', 'calf',
         'camcorder', 'camel', 'camera', 'camera_lens', 'camper_(vehicle)',
         'can', 'can_opener', 'candelabrum', 'candle', 'candle_holder',
         'candy_bar', 'candy_cane', 'walking_cane', 'canister', 'cannon',
         'canoe', 'cantaloup', 'canteen', 'cap_(headwear)', 'bottle_cap',
         'cape', 'cappuccino', 'car_(automobile)', 'railcar_(part_of_a_train)',
         'elevator_car', 'car_battery', 'identity_card', 'card', 'cardigan',
         'cargo_ship', 'carnation', 'horse_carriage', 'carrot', 'tote_bag',
         'cart', 'carton', 'cash_register', 'casserole', 'cassette', 'cast',
         'cat', 'cauliflower', 'caviar', 'cayenne_(spice)', 'CD_player',
         'celery', 'cellular_telephone', 'chain_mail', 'chair',
         'chaise_longue', 'champagne', 'chandelier', 'chap', 'checkbook',
         'checkerboard', 'cherry', 'chessboard',
         'chest_of_drawers_(furniture)', 'chicken_(animal)', 'chicken_wire',
         'chickpea', 'Chihuahua', 'chili_(vegetable)', 'chime', 'chinaware',
         'crisp_(potato_chip)', 'poker_chip', 'chocolate_bar',
         'chocolate_cake', 'chocolate_milk', 'chocolate_mousse', 'choker',
         'chopping_board', 'chopstick', 'Christmas_tree', 'slide', 'cider',
         'cigar_box', 'cigarette', 'cigarette_case', 'cistern', 'clarinet',
         'clasp', 'cleansing_agent', 'clementine', 'clip', 'clipboard',
         'clock', 'clock_tower', 'clothes_hamper', 'clothespin', 'clutch_bag',
         'coaster', 'coat', 'coat_hanger', 'coatrack', 'cock', 'coconut',
         'coffee_filter', 'coffee_maker', 'coffee_table', 'coffeepot', 'coil',
         'coin', 'colander', 'coleslaw', 'coloring_material',
         'combination_lock', 'pacifier', 'comic_book', 'computer_keyboard',
         'concrete_mixer', 'cone', 'control', 'convertible_(automobile)',
         'sofa_bed', 'cookie', 'cookie_jar', 'cooking_utensil',
         'cooler_(for_food)', 'cork_(bottle_plug)', 'corkboard', 'corkscrew',
         'edible_corn', 'cornbread', 'cornet', 'cornice', 'cornmeal', 'corset',
         'romaine_lettuce', 'costume', 'cougar', 'coverall', 'cowbell',
         'cowboy_hat', 'crab_(animal)', 'cracker', 'crape', 'crate', 'crayon',
         'cream_pitcher', 'credit_card', 'crescent_roll', 'crib', 'crock_pot',
         'crossbar', 'crouton', 'crow', 'crown', 'crucifix', 'cruise_ship',
         'police_cruiser', 'crumb', 'crutch', 'cub_(animal)', 'cube',
         'cucumber', 'cufflink', 'cup', 'trophy_cup', 'cupcake', 'hair_curler',
         'curling_iron', 'curtain', 'cushion', 'custard', 'cutting_tool',
         'cylinder', 'cymbal', 'dachshund', 'dagger', 'dartboard',
         'date_(fruit)', 'deck_chair', 'deer', 'dental_floss', 'desk',
         'detergent', 'diaper', 'diary', 'die', 'dinghy', 'dining_table',
         'tux', 'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher',
         'dishwasher_detergent', 'diskette', 'dispenser', 'Dixie_cup', 'dog',
         'dog_collar', 'doll', 'dollar', 'dolphin', 'domestic_ass', 'eye_mask',
         'doorbell', 'doorknob', 'doormat', 'doughnut', 'dove', 'dragonfly',
         'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit',
         'dresser', 'drill', 'drinking_fountain', 'drone', 'dropper',
         'drum_(musical_instrument)', 'drumstick', 'duck', 'duckling',
         'duct_tape', 'duffel_bag', 'dumbbell', 'dumpster', 'dustpan',
         'Dutch_oven', 'eagle', 'earphone', 'earplug', 'earring', 'easel',
         'eclair', 'eel', 'egg', 'egg_roll', 'egg_yolk', 'eggbeater',
         'eggplant', 'electric_chair', 'refrigerator', 'elephant', 'elk',
         'envelope', 'eraser', 'escargot', 'eyepatch', 'falcon', 'fan',
         'faucet', 'fedora', 'ferret', 'Ferris_wheel', 'ferry', 'fig_(fruit)',
         'fighter_jet', 'figurine', 'file_cabinet', 'file_(tool)',
         'fire_alarm', 'fire_engine', 'fire_extinguisher', 'fire_hose',
         'fireplace', 'fireplug', 'fish', 'fish_(food)', 'fishbowl',
         'fishing_boat', 'fishing_rod', 'flag', 'flagpole', 'flamingo',
         'flannel', 'flash', 'flashlight', 'fleece', 'flip-flop_(sandal)',
         'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal',
         'folding_chair', 'food_processor', 'football_(American)',
         'football_helmet', 'footstool', 'fork', 'forklift', 'freight_car',
         'French_toast', 'freshener', 'frisbee', 'frog', 'fruit_juice',
         'fruit_salad', 'frying_pan', 'fudge', 'funnel', 'futon', 'gag',
         'garbage', 'garbage_truck', 'garden_hose', 'gargle', 'gargoyle',
         'garlic', 'gasmask', 'gazelle', 'gelatin', 'gemstone', 'giant_panda',
         'gift_wrap', 'ginger', 'giraffe', 'cincture',
         'glass_(drink_container)', 'globe', 'glove', 'goat', 'goggles',
         'goldfish', 'golf_club', 'golfcart', 'gondola_(boat)', 'goose',
         'gorilla', 'gourd', 'surgical_gown', 'grape', 'grasshopper', 'grater',
         'gravestone', 'gravy_boat', 'green_bean', 'green_onion', 'griddle',
         'grillroom', 'grinder_(tool)', 'grits', 'grizzly', 'grocery_bag',
         'guacamole', 'guitar', 'gull', 'gun', 'hair_spray', 'hairbrush',
         'hairnet', 'hairpin', 'ham', 'hamburger', 'hammer', 'hammock',
         'hamper', 'hamster', 'hair_dryer', 'hand_glass', 'hand_towel',
         'handcart', 'handcuff', 'handkerchief', 'handle', 'handsaw',
         'hardback_book', 'harmonium', 'hat', 'hatbox', 'hatch', 'veil',
         'headband', 'headboard', 'headlight', 'headscarf', 'headset',
         'headstall_(for_horses)', 'hearing_aid', 'heart', 'heater',
         'helicopter', 'helmet', 'heron', 'highchair', 'hinge', 'hippopotamus',
         'hockey_stick', 'hog', 'home_plate_(baseball)', 'honey', 'fume_hood',
         'hook', 'horse', 'hose', 'hot-air_balloon', 'hotplate', 'hot_sauce',
         'hourglass', 'houseboat', 'hummingbird', 'hummus', 'polar_bear',
         'icecream', 'popsicle', 'ice_maker', 'ice_pack', 'ice_skate',
         'ice_tea', 'igniter', 'incense', 'inhaler', 'iPod',
         'iron_(for_clothing)', 'ironing_board', 'jacket', 'jam', 'jean',
         'jeep', 'jelly_bean', 'jersey', 'jet_plane', 'jewelry', 'joystick',
         'jumpsuit', 'kayak', 'keg', 'kennel', 'kettle', 'key', 'keycard',
         'kilt', 'kimono', 'kitchen_sink', 'kitchen_table', 'kite', 'kitten',
         'kiwi_fruit', 'knee_pad', 'knife', 'knight_(chess_piece)',
         'knitting_needle', 'knob', 'knocker_(on_a_door)', 'koala', 'lab_coat',
         'ladder', 'ladle', 'ladybug', 'lamb_(animal)', 'lamb-chop', 'lamp',
         'lamppost', 'lampshade', 'lantern', 'lanyard', 'laptop_computer',
         'lasagna', 'latch', 'lawn_mower', 'leather', 'legging_(clothing)',
         'Lego', 'lemon', 'lemonade', 'lettuce', 'license_plate', 'life_buoy',
         'life_jacket', 'lightbulb', 'lightning_rod', 'lime', 'limousine',
         'linen_paper', 'lion', 'lip_balm', 'lipstick', 'liquor', 'lizard',
         'Loafer_(type_of_shoe)', 'log', 'lollipop', 'lotion',
         'speaker_(stereo_equipment)', 'loveseat', 'machine_gun', 'magazine',
         'magnet', 'mail_slot', 'mailbox_(at_home)', 'mallet', 'mammoth',
         'mandarin_orange', 'manger', 'manhole', 'map', 'marker', 'martini',
         'mascot', 'mashed_potato', 'masher', 'mask', 'mast',
         'mat_(gym_equipment)', 'matchbox', 'mattress', 'measuring_cup',
         'measuring_stick', 'meatball', 'medicine', 'melon', 'microphone',
         'microscope', 'microwave_oven', 'milestone', 'milk', 'minivan',
         'mint_candy', 'mirror', 'mitten', 'mixer_(kitchen_tool)', 'money',
         'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor',
         'motor_scooter', 'motor_vehicle', 'motorboat', 'motorcycle',
         'mound_(baseball)', 'mouse_(animal_rodent)',
         'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom',
         'music_stool', 'musical_instrument', 'nailfile', 'nameplate',
         'napkin', 'neckerchief', 'necklace', 'necktie', 'needle', 'nest',
         'newsstand', 'nightshirt', 'nosebag_(for_animals)',
         'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'nutcracker',
         'oar', 'octopus_(food)', 'octopus_(animal)', 'oil_lamp', 'olive_oil',
         'omelet', 'onion', 'orange_(fruit)', 'orange_juice', 'oregano',
         'ostrich', 'ottoman', 'overalls_(clothing)', 'owl', 'packet',
         'inkpad', 'pad', 'paddle', 'padlock', 'paintbox', 'paintbrush',
         'painting', 'pajamas', 'palette', 'pan_(for_cooking)',
         'pan_(metal_container)', 'pancake', 'pantyhose', 'papaya',
         'paperclip', 'paper_plate', 'paper_towel', 'paperback_book',
         'paperweight', 'parachute', 'parakeet', 'parasail_(sports)',
         'parchment', 'parka', 'parking_meter', 'parrot',
         'passenger_car_(part_of_a_train)', 'passenger_ship', 'passport',
         'pastry', 'patty_(food)', 'pea_(food)', 'peach', 'peanut_butter',
         'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'pegboard',
         'pelican', 'pen', 'pencil', 'pencil_box', 'pencil_sharpener',
         'pendulum', 'penguin', 'pennant', 'penny_(coin)', 'pepper',
         'pepper_mill', 'perfume', 'persimmon', 'baby', 'pet', 'petfood',
         'pew_(church_bench)', 'phonebook', 'phonograph_record', 'piano',
         'pickle', 'pickup_truck', 'pie', 'pigeon', 'piggy_bank', 'pillow',
         'pin_(non_jewelry)', 'pineapple', 'pinecone', 'ping-pong_ball',
         'pinwheel', 'tobacco_pipe', 'pipe', 'pistol', 'pita_(bread)',
         'pitcher_(vessel_for_liquid)', 'pitchfork', 'pizza', 'place_mat',
         'plate', 'platter', 'playing_card', 'playpen', 'pliers',
         'plow_(farm_equipment)', 'pocket_watch', 'pocketknife',
         'poker_(fire_stirring_tool)', 'pole', 'police_van', 'polo_shirt',
         'poncho', 'pony', 'pool_table', 'pop_(soda)', 'portrait',
         'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot',
         'potato', 'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn',
         'printer', 'projectile_(weapon)', 'projector', 'propeller', 'prune',
         'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'pumpkin', 'puncher',
         'puppet', 'puppy', 'quesadilla', 'quiche', 'quilt', 'rabbit',
         'race_car', 'racket', 'radar', 'radiator', 'radio_receiver', 'radish',
         'raft', 'rag_doll', 'raincoat', 'ram_(animal)', 'raspberry', 'rat',
         'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt',
         'recliner', 'record_player', 'red_cabbage', 'reflector',
         'remote_control', 'rhinoceros', 'rib_(food)', 'rifle', 'ring',
         'river_boat', 'road_map', 'robe', 'rocking_chair', 'roller_skate',
         'Rollerblade', 'rolling_pin', 'root_beer',
         'router_(computer_equipment)', 'rubber_band', 'runner_(carpet)',
         'plastic_bag', 'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag',
         'safety_pin', 'sail', 'salad', 'salad_plate', 'salami',
         'salmon_(fish)', 'salmon_(food)', 'salsa', 'saltshaker',
         'sandal_(type_of_shoe)', 'sandwich', 'satchel', 'saucepan', 'saucer',
         'sausage', 'sawhorse', 'saxophone', 'scale_(measuring_instrument)',
         'scarecrow', 'scarf', 'school_bus', 'scissors', 'scoreboard',
         'scrambled_eggs', 'scraper', 'scratcher', 'screwdriver',
         'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seaplane',
         'seashell', 'seedling', 'serving_dish', 'sewing_machine', 'shaker',
         'shampoo', 'shark', 'sharpener', 'Sharpie', 'shaver_(electric)',
         'shaving_cream', 'shawl', 'shears', 'sheep', 'shepherd_dog',
         'sherbert', 'shield', 'shirt', 'shoe', 'shopping_bag',
         'shopping_cart', 'short_pants', 'shot_glass', 'shoulder_bag',
         'shovel', 'shower_head', 'shower_curtain', 'shredder_(for_paper)',
         'sieve', 'signboard', 'silo', 'sink', 'skateboard', 'skewer', 'ski',
         'ski_boot', 'ski_parka', 'ski_pole', 'skirt', 'sled', 'sleeping_bag',
         'sling_(bandage)', 'slipper_(footwear)', 'smoothie', 'snake',
         'snowboard', 'snowman', 'snowmobile', 'soap', 'soccer_ball', 'sock',
         'soda_fountain', 'carbonated_water', 'sofa', 'softball',
         'solar_array', 'sombrero', 'soup', 'soup_bowl', 'soupspoon',
         'sour_cream', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)',
         'spatula', 'spear', 'spectacles', 'spice_rack', 'spider', 'sponge',
         'spoon', 'sportswear', 'spotlight', 'squirrel',
         'stapler_(stapling_machine)', 'starfish', 'statue_(sculpture)',
         'steak_(food)', 'steak_knife', 'steamer_(kitchen_appliance)',
         'steering_wheel', 'stencil', 'stepladder', 'step_stool',
         'stereo_(sound_system)', 'stew', 'stirrer', 'stirrup',
         'stockings_(leg_wear)', 'stool', 'stop_sign', 'brake_light', 'stove',
         'strainer', 'strap', 'straw_(for_drinking)', 'strawberry',
         'street_sign', 'streetlight', 'string_cheese', 'stylus', 'subwoofer',
         'sugar_bowl', 'sugarcane_(plant)', 'suit_(clothing)', 'sunflower',
         'sunglasses', 'sunhat', 'sunscreen', 'surfboard', 'sushi', 'mop',
         'sweat_pants', 'sweatband', 'sweater', 'sweatshirt', 'sweet_potato',
         'swimsuit', 'sword', 'syringe', 'Tabasco_sauce', 'table-tennis_table',
         'table', 'table_lamp', 'tablecloth', 'tachometer', 'taco', 'tag',
         'taillight', 'tambourine', 'army_tank', 'tank_(storage_vessel)',
         'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure',
         'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup',
         'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth',
         'telephone_pole', 'telephoto_lens', 'television_camera',
         'television_set', 'tennis_ball', 'tennis_racket', 'tequila',
         'thermometer', 'thermos_bottle', 'thermostat', 'thimble', 'thread',
         'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer',
         'tinfoil', 'tinsel', 'tissue_paper', 'toast_(food)', 'toaster',
         'toaster_oven', 'toilet', 'toilet_tissue', 'tomato', 'tongs',
         'toolbox', 'toothbrush', 'toothpaste', 'toothpick', 'cover',
         'tortilla', 'tow_truck', 'towel', 'towel_rack', 'toy',
         'tractor_(farm_equipment)', 'traffic_light', 'dirt_bike',
         'trailer_truck', 'train_(railroad_vehicle)', 'trampoline', 'tray',
         'tree_house', 'trench_coat', 'triangle_(musical_instrument)',
         'tricycle', 'tripod', 'trousers', 'truck', 'truffle_(chocolate)',
         'trunk', 'vat', 'turban', 'turkey_(bird)', 'turkey_(food)', 'turnip',
         'turtle', 'turtleneck_(clothing)', 'typewriter', 'umbrella',
         'underwear', 'unicycle', 'urinal', 'urn', 'vacuum_cleaner', 'valve',
         'vase', 'vending_machine', 'vent', 'videotape', 'vinegar', 'violin',
         'vodka', 'volleyball', 'vulture', 'waffle', 'waffle_iron', 'wagon',
         'wagon_wheel', 'walking_stick', 'wall_clock', 'wall_socket', 'wallet',
         'walrus', 'wardrobe', 'wasabi', 'automatic_washer', 'watch',
         'water_bottle', 'water_cooler', 'water_faucet', 'water_filter',
         'water_heater', 'water_jug', 'water_gun', 'water_scooter',
         'water_ski', 'water_tower', 'watering_can', 'watermelon',
         'weathervane', 'webcam', 'wedding_cake', 'wedding_ring', 'wet_suit',
         'wheel', 'wheelchair', 'whipped_cream', 'whiskey', 'whistle', 'wick',
         'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)',
         'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket',
         'wineglass', 'wing_chair', 'blinder_(for_horses)', 'wok', 'wolf',
         'wooden_spoon', 'wreath', 'wrench', 'wristband', 'wristlet', 'yacht',
         'yak', 'yogurt', 'yoke_(animal_equipment)', 'zebra', 'zucchini',
         # v0_new_caption
         'circle', 'grapes', 'moon', 'flower', 'wood', 'car', 'couch', 'referee', 'tower', 'bus', 'square', 'door', 'biscuit', 'barcode', 'lotion', 'tennis_player', 'man', 'fire_hydrant', 'cork', 'cloth', 'rock', 'coal', 'train', 'house', 'cactus', 'rocket', 'cell_phone', 'hand', 'stick', 'window', 'plant', 'ant', 'machine', 'matchstick', 'bathroom', 'lung', 'chicken', 'chip', 'chips', 'stone', 'leaf', 'child', 'leaves', 'face', 'axe', 'statue', 'bone', 'worm', 'hole', 'board', 'bag', 'qr_code', 'paper', 'donut', 'lodge', 'restaurant', 'star', 'letter', 'motel', 'fridge', 'baseball_player', 'pants', 'keyboard', 'soda', 'soccer_player', 'goalie', 'baby', 'mustache', 'arm', 'soccer', 'human', 'leg', 'tie', 'chili', 'mouse', 'phone', 'game', 'turbocharger', 'lighthouse', 'satellite', 'cross', 'roof', 'plane', 'bridge', 'television', 'ship', 'seesaw', 'meat', 'catcher', 'pipeline', 'stairs', 'triangle', 'crystal', 'penis', 'hot_dog', 'scooter', 'math_book', 'eye', 'vegetable', 'glass', 'exit', 'drink', 'torch', 'brush', 'nose', 'tooth', 'rail', 'brick', 'computer', 'branch', 'mole', 'speed_limit', 'rugby_player', 'jet', 'skier', 'garden_tool', 'rectangle', 'band_saw', 'skateboarder', 'castle', 'arch', 'rug', 'bathroom_sink', 'finger', 'tractor', 'line', 'scythe', 'match', 'track', 'chopsticks', 'nail', 'laptop', 'blinds', 'lighter', 'mannequin', 'bat', 'tv_stand', 'robot', 'batter', 'column', 'football_player', 'microwave', 'plug', 'pet_food', 'wine_glass', 'pan', 'food', 'christmas_decoration', 'chocolate', 'shelf', 'chain', 'rooster', 'gate', 'tank', 'gas_pump', 'sign', 'church', 'scale', 'light', 'drum', 'light_bulb', 'loading_bar', 'shower', 'cupcake_pan', 'flute', 'tea_pot', 'tape', 'tile', 'animal', 'adapter', 'computer_mouse', 'wire', 'river', 'yonex', 'garbage_bag', 'smoke', 'fence', 'crossbow', 'nuts', 'breast', 'orange', 'boy', 'dragon', 'skis', 'caterpillar', 'light_switch', 'golf_player', 'tablet', 'toilet_paper', 'footprint', 'ice_cream', 'noose', 'fly', 'hair', 'string', 'onigiri', 'dairy_queen', 'monitor', 'cable', 'price_tag', 'plum', 'package', 'bath', 'bookshelf', 'octopus', 'forest', 'frame', 'metal', 'peanuts', 'peanut', 'carpet', 'safe', 'potato_chip', 'wedgwood', 'trophie', 'fern', 'cotton', 'no_parking', 'bee', 'roast', 'fire', 'woman', 'firework', 'chest', 'churro', 'champagne', 'fruit', 'police_officer', 'bush', 'wing', 'vegetables', 'shell', 'popcorn', 'purse', 'coral', 'rainbow', 'cheese', 'chameleon', 'tent', 'planet', 'football', 'brain', 'snail', 'lightning', 'traffic_sign', 'gas_station', 'coca-cola', 'pill', 'whale', 'cans', 'broken_glass', 'wheelbarrow', 'burger', 'jellyfish', 'arrow', 'usb', 'speaker', 'file', 'entertainment_center', 'horn', 'elevator', 'rhino', 'snowflake', 'stripes', 'paint', 'pistachio', 'corn', 'curve', 'cord', 'dot', 'wand', 'satellite_dish', 'document', 'knight', 'film', 'moth', "spider-man's_webbing", 'contact_lens', 'store', 'tornado', 'spice', 'panda', 'sea_horse', 'tongue', 'net', 'credit_card', 'server', 'disk', 'salt_shaker', 'wave', 'library', 'ratchet', 'suit', 'crab', 'steak', 'website', 'leek', 'rose', 'crocodile', 'cloud', 'ram', 'remote', 'string_lights', 'stair', 'kiwi', 'squash', 'trophy', 'tennis_court', 'lamb', 'donkey', 'cabbage', 'number', 'bean', 'traffic_cone', 'pipette', 'laser_pointer', 'mayonnaise', 'olive', 'bus_stop', 'nugget', 'root', 'parking', 'paint_can', 'zoo', 'rocks', 'crane', 'comb', 'straw', 'cardboard', 'breakfast', 'sun', 'foot', 'sphere', 'spiral', 'anvil', 'weight', 'swing', 'fruits', 'elbow', 'coffee_cup', 'coffee', 'bank', 'peacock', 'head', 'ticket', 'antelope', 'cell', 'luggage', 'barrier', 'golf_hole', 'rice', 'note', 'skull', 'crank', 'taxi', 'ricotta_cheese', 'container', 'well', 'tire', 'id', 'toiletries', 'ipod', "mcdonald's", 'bug', 'motorhome', 'feather', 'oakley', 'tempura', 'dinosaur', 'pitcher', 'diamond', 'tea_kettle', 'lacoste', 'mp3_player', 'barn', 'christmas_stocking', 'bacon', 'picture', 'telescope', 'fries', 'ufo', 'character', 'shaver', 'baseball_jersey', 'bison', 'glasses', 'space_station', 'closet', 'highway', 'sailboat', 'shrimp', 'teeth', 'police', 'pizza_cutter', 'report', 'snowblower', 'soap_dispenser', 'bike', 'ramp', 'farm', 'shoes', 'batsman', 'filing_cabinet', 'console', 'hard_drive', 'controller', 'surfer', 'juice', 'shorts', 'beer', 'coffee_filter', 'skate', 'parsnip', 'turbo', 'spray_bottle', 'insect', 'hair_spray', 'game_controller', 'paper_bag', 'ocean', 'fingerprint', 'socks', 't-shirt', 'ear', 'mac', 'swan', 'washing_machine', 'golf_ball', 'candy', 'radio', 'garden', 'trombone', 'turkey', 'dancer', 'garbage_can', 'gas_canister', 'moss', 'snow', 'alphabet', 'bagpipes', 'granite', 'blob', 'beach', 'train_track', 'condom', 'van', 'beans', 'chisel', 'trailer', 'washer', 'tissue', 'flags', 'beam', 'palace', 'pillar', 'whisker', 'deodorant', 'stack', 'beehive', 'mathematics', 'panther', 'macaron', 'emoji', 'headphones', 'spade', 'pendant', 'anchor', 'energy_drink', 'baseboard', 'totem', 'usb_cable', 'plant_pot', 'continent', 'lipstick', 'submarine', 'island', 'lock', 'tissue_dispenser', 'hand_dryer', 'pirate', 'pasta', 'saw', 'airport', 'badger', 'thumb', 'trolley', 'city', 'land', 'wine', 'boulder', 'intercom', 'kangaroo', 'eiffel_tower', 'garage', 'marikareinvented', 'bunny', 'harp', 'cpu', 'hedgehog', 'wii_remote', 'computer_chip', 'bunker', 'spaghetti', 'bird_feeder', 'buffalo', 'curved_object', 'chess_piece', 'sea', 'cd_player', 'cd', 'id_card', 'jail', 'power_cord', 'starbucks', 'logo', 'basketball_court', 'spacecraft', 'bill', 'level', 'router', 'ice', 'puck', 'dessert', 'cricket_player', 'squid', 'gooseberry', 'spray_can', 'strip', 'biker', 'spray', 'gel', 'granny_smith', 'golf', 'gas_meter', 'soldier', 'weapon', 'train_station', 'dice', 'bow', 'lamp_post', 'atm', 'seaweed', 'gavel', 'flowers', 'bud', 'hand_sanitizer', 'computer_monitor', 'girl', 'speed_limit_sign', 'wood_stove', 'network_cable', 'curved', 'exclamation_mark', 'emirates', 'game_console', 'wedding_dress', 'bouncy_castle', 'rosary', 'beak', 'dart', 'tattoo_shop', 'stack_of_books', 'kitchen', 'scorpion', 'tea_cup', 'blower', 'cartoon_character', 'milk_jug', 'neon', 'neon_sign', 'cardiovascular_system', 'jet_ski', 'waiter', 'rosemary', 'raisins', 'freezer', 'cricketer', 'light_pole', 'rake', 'chamber', 'sock_monkey', 'name_tag', 'sweeper', 'countertop', 'coca_cola', 'toys', 'goblin', 'skeleton', 'snowboarder', 'bullet', 'speedometer', 'surgical_instrument', 'santa_claus', 'drinking_cup', '7_days', 'portrait', 'gold', 'matches', 'guinea_pig', 'oil', 'chef', 'toast', 'garment', 'gnome', 'memory', 'ribs', 'chewing_gum', 'liver', 'fountain', 'leash', 'zipper', 'nightstand', 'highway_sign', 'swimmer', 'stroller', 'flatbread', 'handrail', 'chainsaw', 'hello_kitty', 'lip_gloss', 'jugs', 'moose', 'hedge_trimmer', 'sari', 'pushpin', 'pig', 'long_neck', 'screw', 'tumor', 'movie_poster', 'hexagon', 'fox', 'crockpot', 'usb_drive', 'volleyball_player', 'juul', 'transistor', 'goblet', 'africa', 'parking_sign', 'diver', 'beach_ball', 'sanitary_napkin', 'eggs', 'hot', 'shadow', 'leech', 'pea', 'cashew', 'hair_straightener', 'jeans', 'icicle', 'armored_vehicle', 'barbed_wire', 'stack_of_money', 'wi-fi', 'cooler', 'date', 'wii', 'tv', 'band', 'croissant', 'paint_roller', 'wooden_frame', 'lace', 'toddler', 'ruler', 'postbox', 'wii_console', 'go', 'safety_harness', 'cows', 'apothecary', 'granola', 'fluorescent_light', 'mixer', 'olives', 'hot_air_balloon', 'europe', 'earrings', 'jelly_beans', 'coffee_bean', 'tardis', 'advertising', 'boots', 'foil', 'gas_tank', 'candlestick', 'hay', 'jockey', 'beads', 'gas_can', 'arc', 'walnut', 'strawberries', 'shaving', 'tea_set', 'browser', 'certificate', 'samsung', 'beet', 'unicorn', 'hearing_aid', 'stethoscope', 'lava_lamp', 'cosmetics', 'buddha', 'gazebo', 'motorcycle_rider', 'salt', 'path', 'tennis', 'meter', 'seat_belt', 'gears', 'plastic', 'army', 'curry', 'mouse_pad', 'ski_jumper', 'supermarket', 'socket', 'ruin', 'neck', 'garbage_bin', 'knot', 'mermaid', 'glasses_case', 'baked_beans', 'cartoon', 'vacuum', 'skater', 'obelisk', 'post-it_note', 'lighting_box', 'fly_swatter', 'ferris_wheel', 'jester', 'meerkat', 'paint_palette', 'gatorade', 'wine_cooler', 'silos', 'sailboard', 'cheetah', 'tuk_tuk', 'powder', 'fist', 'jalapeno', 'keyhole', 'speed_camera', 'block', 'graphene', 'guitarist', 'vial', 'carabiner', 'police_uniform', 'jars', 'wooden_ladder', 'stadium', 'text', 'quill', 'cereal', 'leopard', 'scanner', 'rubber', 'spike', 'silhouette', 'rope', 'olympic_rings', 'carriage', 'aircraft', 'bodybuilder', 'scalpel', 'beer_glass', 'bell_tower', 'molecule', 'missile', 'speech_bubble', 'concrete', 'blade', 'temple', 'cat_food', 'pedestrian_crossing', 'door_handle', 'berries', 'subway', 'boxer', 'escalator', 'popsicle_stick', 'nuclear_explosion', 'gum', 'nasa_glenn_research_center', 'toad', 'whiteboard', 'tea', 'grated_cheese', 'antler', 'snowball', 'stahl', 'noodles', 'paint_tube', 'check_mark', 'eyes', 'fringe', 'asteroid', 'rugby_ball', 'pizza_oven', 'engine', 'no_parking_sign', 'curio_cabinet', 'banknote', 'lips', 'iceberg', 'jaw', 'chairs', 'pizza_sauce', 'roller', 'california', 'pyramid', 'iv_bag', 'chart', 'grocery', 'light_saber', 'compressor', 'jet_engine', 'coupon', 'spaceship', 'lake', 'cowboy', 'lama', 'petri_dish', 'bathroom_faucet', 'toilets', 'charcoal', 'grenade', 'state', 'garden_gate', 'chicken_nugget', 'mower', 'bust', 'rim', 'paper_clip', 'parsley', 'kiosk', 'ghost', 'fractal', 'seed', 'gift', 'wooden_object', 'chrysalis', 'golfer', 'noogie', 'risotto', 'wooden_stick', 'grave', 'gear', 'fireworks', 'airship', 'check', 'clothes', 'stomach', 'paper_shredder', 'cigar', 'alien', 'sauce', 'horse_shoe', 'onion_ring', 'hatchet', 'test_tube', 'fish_tank', 'graph', 'duffle_bag', 'sheet_music', 'database', 'collar', 'tomato_sauce', 'wine_rack', 'basketball_post', 'beaker', 'menu', 'salon', 'lens', 'dvd_player', 'pickaxe', 'gingerbread_man', 'puzzle', 'paper_airplane', 'porch', 'record', 'gas_stove', 'skunk', 'comic', 'minion', 'salt_and_pepper_shakers', 'school_supplies', 'cutlery', 'blackberries', 'mammal', 'iron', 'bench_grinder', 'armour', 'espn', 'solar_panel', 'truffle', 'arcade_machine', 'laser', 'light_tower', 'cat_scratcher', 'christmas_ornament', 'uniform', 'drum_set', 'armadillo', 'whiskey', 'hotdog', 'dirt', 'park', 'swimming_pool', 'hoodie', 'picture_frame', 'charger', 'bunch_of_asparagus', 'sketch', 'receiver', 'starship', 'grip', 'porcupine', 'shark_tooth', 'excavator', 'trowel', 'mortar', 'trash_bag', 'walkers', 'hedge', 'handicap', 'table_tennis_racket', 'hospital', 'advocate_health_care', 'fire_truck', 'hockey_player', 'abacus', 'cyclist', 'beef', 'rover', 'toilet_brush', 'usb_flash_drive', 'kitchen_cabinet', 'traffic_fines', 'nutribullet', 'power_line', 'atom', 'toffee', 'basketball_hoop', 'wetsuit', 'chicken_leg', 'coins', 'hanging_object', 'tunnel', 'bale', 'tobacco', 'laser_scanner', 'ornament', 'footwear', 'eeyore', 'henna', 'stapler', 'skin', 'town_center', 'tool_box', 'tape_player', 'construction_worker', 'geico', 'wooden_crate', 'coffee_machine', 'sphinx', 'hot_dog_bun', 'art', 'garden_fence', 'construction', 'cannon', 'peas', 'basketball_goal', 'game_button', 'christ', 'cave', 'chest_of_drawers', 'grain', 'motorcycle_racer', 'trumpet', 'bar', 'matte', 'dip', 'hair_clip', 'cocktail', 'ketchup_bottle', 'cricket', 'kiteboarder', 'post-it', 'cream', 'knee', 'goal', 'green_beans', 'mustard', 'macaroni', 'tube', 'camera_bag', 'ceiling', 'airplane_seat', 'juicer', 'yak', 'lot', 'game_character', 'french_fries', 'osterizer', 'origami', 'beijing', 'seedling', 'burnt', 'party_horn', 'party_hat', 'corn_dog', 'picnic', 'nuclear_power_plant', 'checkered', 'cork_screw', 'mat', 'warehouse', 'blowpipe', 'security_camera', 'monster', 'cane', 'feet', 'granade', 'music_note', 'virus', 'glue_gun', 'cleaner', 'clay', 'utopia', 'utensil', 'mobile_phone', 'guard', 'dj_mixer', 'curved_surface', 'skateboard_ramp', 'tail', 'kidney', 'altoids', 'ribbon', 'bed_sheet', 'chain_saw', 'nasa', 'american_flag', 'bacteria', 'buttons', 'rod', 'scroll', 'casket', 'slippers', 'coast_guard', 'door_hinge', 'graffiti', 'audio_equipment', 'tool', 'cowgirl', 'mummy', 'adidas', 'lobster', 'peanut_brittle', 'plastic_bottle', 'oyster', 'pepsi', 'wooden_fence', 'brow_gel', 'joker', 'galaxy', 'gun_case', 'clown', 'champagne_glass', 'currency', 'electronic_device', 'caliper', 'train_car', 'smartphone', 'sublet', 'hump', 'pedometer', 'pocket', 'name_card', 'pin', 'badminton_racket', 'bump', 'soybeans', 'phone_booth', 'g-clamp', 'baseball_catcher', 'pipelines', 'octagon', 'centipede', 'tortilla_chip', 'shoelace', 'harmonica', 'zombie', 'country', 'lifter', 'samurai', 'hot_tub', 'bomb', 'chain_link_fence', 'canned_food', 'razor', 'capitol', 'baseball_helmet', 'salmon', 'carpaccio', 'laryngoscope', 'gutter', 'soap_bottle', 'hotel', 'otter', 'dock', 'drain', 'toledo', 'jigsaw_puzzle', 'gas_bottle', 'chestnut', 'film_reel', 'birthday_candle', 'transformer', 'rain', 'capsule', 'cap', 'microphone_stand', 'dehumidifier', 'blocks', 'start', 'peppercorns', 'sea_slug', 'progress_bar', 'fiber_optic_cable', 'silicone', 'cable_car', 'cd-rom_drive', 'aluminum_foil', 'landscape', 'audio_mixer', 'gpu', 'superhero', 'bowtie', 'pillbox', 'wine_barrel', 'one_way', 'astronaut', 'mold', 'paintball', 'sound_mixer', 'mixing_console', 'basketball_jersey', 'podium', 'school_crossing', 'school', 'slazenger', 'drinking_glass', 'sheet', 'striped', 'diffuser', 'hand_grenade', 'okra', 'scaffolding', 'mosque', 'implant', 'franchise', 'staged', 'wind_turbine', 'tattoo', 'milk_bottle', 'cricket_bat', 'google_logo', 'pulpit', 'one-way_sign', 'amphibious_vehicle', 'exit_sign', 'bandana', 'counter', 'shredder', 'toilet_flush_valve', 'luggage_cart', 'factory', 'folder', 'fishnet', 'xfinity', 'lumberjack', 'cubicle', 'gun_magazine', 'bird_feather', 'red_ball', 'stormtrooper', 'arcade_game', 'fire_escape', 'mantle', 'bangle', 'gecko', 'body_armor', 'tamarind', 'poppy', 'clothing', 'pedal', 'basketball_player', 'oil_rig', 'antarctica', 'dna', 'tubing', 'arrowhead', 'soda_can', 'seal', 'stick_figure', 'corner', 'currency_exchange', 'lily', 'thrasher', 'chafing_dish', 'lower_third', 'plunger', 'walkway', 'teabag', 'crust', 'glitter', 'power_supply_unit', 'network', 'chimney', 'organ', 'resume', 'barbecue', 'granule', 'window_squeegee', 'aeroshell', 'pump', 'tampon', 'boxing_bag', 'world', 'portable_toilet', 'gunman', 'moai', 'name', 'lost_&_found', 'cornflakes', 'balance', 'doctor', 'ketchup', 'ray', 'larva', 'tub', 'catheter', 'tulip', 'hawk', 'ski_lift', 'float', 'dental_instrument', 'ibm', 'hanger', 'bean_sprouts', 'bar_stool', 'crossword_puzzle', 'haircut', 'olympics', 'grate', 'cups', 'zipper_bag', 'crates', 'washington_monument', 'quad_bike', 'rv', 'wireframe', 'fuse_box', 'bong', 'belmont', 'oat', 'tweezers', 'pancakes', 'diapers', 'food_truck', 'space_needle', 'staircase', 'power_station', 'nameplate', 'coriander', 'pebbles', 'detour', 'cutting_board', 'space', 'mouth', 'thermos', 'conveyor_belt', 'no_object', 'gasket', 'boar', 'bean_bag', 'baldness', 'fracture', 'cornhole', 'slot_machine', 'manta_ray', 'hang_glider', 'firefighter', 'mace', 'printers', 'pedestrian', 'granola_bar', 'boomerang', 'sleeve', 'goalpost', 'butternut_squash', 'atv', 'power_adapter', 'filter', 'cinnamon', 'acorn', 'fire_exit', 'athlete', 'sake_bottle', 'sake', 'firefighter_jacket', 'climber', 'court', 'jerrycan', 'monk', 'electric_meter', 'digger', 'racism', 'hockey_jersey', 'wrist', 'hairdressing_tools', 'lipsstick', 'fedex', 'marshmallow', 'pallet', 'capacitor', 'wooden_structure', "plumber's_wrench", 't_wrench', 'intestine', 'pack', 'stegosaurus', 'fossil', 'shutter', 'post', 'giant', 'domo', 'whale_tooth', 'jaguar', 'striped_fabric', 'dim_sum', 'motorcyclist', 'hospital_bed', 'fried_rice', 'cranberry', 'lift', 'comet', 'treadmill', 'monument', 'volcano', 'flake', 'slug', 'bear_paw', 'u-shaped', 'cliff', 'palm', 'onions', 'electric_tower', 'skate_ramp', 'guitar_strap', 'egg_tray', 'spring_valley', 'beaver', 'smiley_face', 'connector', 'slab', 'lip', 'marble', 'sperm', 'grapefruit', 'harvester', 'picnic_table', 'baking', 'nuclear_bomb', 'shed', 'firewood', 'books', 'stage', 'pressure_cooker', 'cds', 'milk_carton', 'cheesecake', 'boxing_gloves', 'omelette', 'lindt', 'rogers', 'wooden_block', 'mario', 'dr_pepper', 'bidet', 'oval', 'baseball_uniform', 'copy_paper', 'flying_object', 'railing', 'porridge', 'swoosh', 'whirligig', 'fruit_tart', 'tart', 'product', 'glasshouse', 'hoddie', 'wooden_box', 'onet', 'airbag', 'adhesive_note', 'bathroom_cabinet', 'gong', 'horse_trailer', 'bra', 'recycling_bin', 'hay_bale', 'tea_strainer', 'tap', 'mouthwash', 'golf_bag', 'batman', 'lifeguard', 'cage', 'kelp', 'ceiling_light', 'hot_dog_stand', 'train_signal', 'lungs', 'crumpled_paper', 'film_projector', 'hip', 'shark_fin', 'digit', 'lid', 'tank_top', 'wireless_router', 'stretcher', 'roll_of_paper', 'coffee_beans', 'school_crossing_sign', 'diving_suit', 'drink_can', 'coffee_grinder', 'outhouse', 'target', 'high_jump', 'weather_vane', 'print', 'mailbox', 'nutmeg', 'peeler', 'helium_balloon', 'drinking_straw', 'garage_door', 'reef', 'beard', 'gravy', 'goal_post', 'floatation_device', 'cake_stand', 'police_car', 'processor', 'christmas_wreath', 'coffee_pot', 'nuclear_reactor', 'jiffy_pop', 'potato_salad', 'stonehenge', 'nail_clippers', 'soccer_jersey', 'roller_coaster', 'graduation_cap', 'beverage', 'paw', 'bananas', 'almonds', 'truss', 'chicken_nuggets', 'squeegee', 'wrapping', 'sasquatch', 'raindrop', 'grid', 'painter', 'fish_fin', 'spa', 'jug', 'hilton', 'dental_appliance', 'pressure_washer', 'swimming_trunks', 'tape_dispenser', 'thatched_roof', 'mcdonalds', 'duct', 'pill_bottle', 'video_game', 'saddle', 'a-frame', 'storage', 'fire_pit', 'toilet_paper_holder', 'fur', 'maid', 'meatloaf', 'pikachu', 'plush', 'linkedin_logo', 'gummy_bear', 'crash_test_dummy', 'long-necked_animal', 'cannabis', 'silverware', 'flip_flops', 'lane', 'paint_sprayer', 'tape_recorder', 'artery', 'greenhouse', 'us_open', 'keys', 'printer_ink_cartridges', 'spool', 'whisk', 'pool', 'dinner', 'barriers', 'whirlpool', 'hair_dye', 'stupa', 'back_scratcher', 'medal', 'new_era', 'dollar_sign', 'neuron', 't-rex', 'christmas_lights', 'fang', 'business', 'gas_regulator', 'tick', 'chevrolet_flag', 'ceiling_fan', 'fire_station', 'sugar', 'coach', 'ruby', 'stack_of_coins', 'memory_card', 'blueberries', 'kohlrabi', 'balcony', 'tubular', 'nipple', 'spiderman', 'loom', 'beer_mug', 'board_game', 'floodway', '7-eleven', 'aspen_dental', 'motor_show', 'signpost', 'pond', 'swimming', 'glider', 'nike', 'tabasco', 'surf_rescue', 'plaque', 'reindeer', 'oven_mitt', 'christmas_ball', 'floppy_disk', 'alcohol_bottle', 'belly_button', 'traffic_hazard', 'fetus', 'lava', 'air_mattress', 'furniture', 'cement', 'snow_shovel', 'maracas', 'accordian', 'bollard', 'canned_drink', 'football_jersey', 'menstrual_cup', 'architectural_element', 'gantry_crane', 'altar', 'louis_vuitton', 'screen', 'baking_tray', 'berry', 'pecan', 'television_stand', 'earphones', 'dental_surgeon', 'mirror_ball', 'shoe_polish', 'garage_sale', 'aluminum', 'lotus', 'mango', 'yam', 'gummy_bears', 'stuffed_animal', 'french_fry', 'sticker', 'yarn', 'sugar_dispenser', 'fighter', 'flash_drive', 'curved_line', 'invoice', 'blow_dryer', 'angel', 'taj_mahal', 'saree', 'trophies', 'sea_urchin', 'power_strip', 'apple_juice', 'police_badge', 'wind', 'ship_wheel', 'pub', 'sparkler', 'baking_pan', 'spinach', 'protein', 'bunch_of_grapes', 'cable_tie', 'soap_dish', 'gas_station_sign', 'shop', 'segway', 'no_entry', 'walkie_talkie', 'power_supply', 'circuit', 'armored_suit', 'restroom', 'baguette', 'wine_shop', 'ankle_bracelet', 'trash', 'shells', 'pebble', 'vein', 'bath_tub', 'yamaha', 'cutting-board', 'desk_assistants', 'baker', 'movie_reel', 'chinese_dumpling', 'whale_tail', 'cash_desk', 'ravioli', 'circular', 'sanitizer', 'mecha', 'soja', 'merrill_lynch_wealth_management', 'laser_sword', 'quinoa', 'curio', 'kale', 'stand', 'network_card', 'math', 'yield_sign', 'jetpack', 'weed', 'indian_takeaway', 'copper', 'piston', 'apples', 'hockey', 'clothesline', 'cd_rack', 'croquet_mallet', 'armrest', 'candelabra', 'cancer', 'scissor_lift', 'queue', 'resistor', 'pepperoni', 'crack', 'arm_band', 'rice_krispies_treats', 'saloon', 'internet', 'hockey_puck', 'pomegranate', 'long_necked_animal', 'bagger', 'robot_arm', 'colgate', 'hay_feeder', 'legs', 'competition_entries', 'corn_meal', 'under_armour', 'trench', 'music_sheet', 'card_game', 'sew', 'sim_card', 'v-neck', 'c', 'yield', 'anteater', 'jump_rope', 'photo', 'muscle', 'cotton_candy', 'explosion', 'csa', 'grocery_store', 'barbecue_sauce', 'measuring_spoon', 'peppers', 'swimming_fin', 'grader', 'snowplow', 'desk_lamp', 'bombe', 'oatmeal', 'hydrant', 'olympic_ring', 'barber_pole', 'jagermeister', 'flour', 'pan_flute', 'sleeveless_shirt', 'lamps', 'swatter', 'automobile', 'serving_spoon', 'dumpling', 'potatoes', 'traffic_officer', 'cooking', 'orchid', 'mascara', 'jerry_can', 'wedding', 'skid_steer', 'golf_cart', 'blood_cell', 'peace_sign', 'construction_equipment', 'horseshoe', 'knee_brace', 'hut', 'bitcoin', 'sippy_cup', 'roll', 'wool', 'navel', 'aeroplane', 'ash', 'hanging_rack', 'carousel', 'poop', 'advertisement', 'pot_lid', 'card_catalog', 'skiier', 'injector', 'fluorescent_light_fixture', 'dish_soap', 'domino', 'drawing', 'badminton_player', 'welsh_dragon', 'bunch_of_vegetables', 'chariot', 'oysters', 'lone_star', 'figure_skater', 'car_seat', 'barbie', 'test-tube', 'trident', 'pop_rocks', 'vhs_player', 'sapphire', 'biplane', 'air_filter', 'sunlounger', 'bass_clef', 'tomato_juice', 'people', 'silk_worm', 'ap', 'tortoise', 'microchip', 'crushed_pineapple', 'hot_fudge', 'fiber', 'music_player', 'fuse', 'budweiser', 'hops', 'wound', 'sea_shell', 'watchtower', 'lemur', 'lego', 'samovar', 'gas_heater', 'jumper', 'pixel_character', 'light_stand', 'merino', 'aqueduct', 'rolex', 'table_tennis_paddle', "m&m's", 'pawn_shop', 'kraft', 'mosquito', 'table_tennis_player', 'badge', 'muppet', 'squeezer', 'cell_tower', 'first_aid_kit', 'parking_lot', 'cafe', 'bottlecap', 'reed', 'eggs_benedict', 'banana_leaf', 'nurse', 'sailor', 'entrance', 'ski_lodge', 'university', 'roulette', 'shaving_brush', 'bowling_alley', 'torii', 'crossing', 'torah', 'video_player', 'wipes', 'extension_cord', 'sieve', 'ikea', 'punch', 'eye_shadow', 'paintball_player', 'wheeled_vehicle', 'kayak_paddle', 'cronut', 'laser_level', 'warning_sign', 'witch', 'blood', 'flower_pot', 'spindle', 'diode', 'bus_station', '3-way', 'roti', 'bikini', 'club', 'enchanter', 'bearded_dragon', 'quad', 't-shaped_object', 'falafel', 'outfit', 'praying_mantis', 'whisper', 'lily_pad', 'syrup', 'bubble', 'ups', 'red_envelope', 'lunchbox', 'dirt_bin', 'fry', 'crescent', 'cutting_mat', 'ox', 'party_popper', 'football_shirt', 'plastic_container', 'furbby', 'grinder', 'vhs_tape', 'gps', 'patchwork', 'durian', 'leatherback_turtle', 'frango', 'presser', 'vape', 'state_farm', 'credit_card_reader', 'lipsense', 'crackers', 'mail', 'music_stand', 'coca_cola_crate', 'vascular', 'bookend', 'geode', 'lays', 'louis', 'play_button', 'lucky_charms', 'parking_ticket', 'hand_truck', 'thumbs_up', 'citizen', 'school_sign', 'dictionary', 'information', 'stick_insect', 'automated_teller_machine', 'cookies', 'christmas_carol', 'fax_machine', 'tomato_soup', 'table_tennis_table', 'calamari', "rubik's_cube", 'album', 'scallop', 'gas_mask', 'cinnamon_roll', 'muffin_tray', 'millstone', 'burn', 'medical', 'voodoo_doll', 'cotton_swab', 'coat_of_arms', 'mosaic', 'flip-flop', 'milk_crate', 'boba_fett', 'wooden_bars', 'pizza_box', 'ouija_board', 'tinkerbell', 'teal', 'stitch', 'brace', 'lazertag', 'salt_mill', 'arms', 'bowling', 'gym', 'spine', 'balls', 'lentil', 'cuckoo_clock', 'airpods', 'political_sign', 'catapult', 'christmas_present', 'donuts', 'balloons', 'dots', 'racecourse', 'fishhook', 'dandelion', 'gymnast', 'tater_tots', 'table_leg', 'baked_goods', 'bowling_pin', 'home', '7_eleven', 'lotto', 'coffee_shop', 'rice_cooker', 'hiking_pole', 'trencher', 'garland', 'hoverboard', 'birthday_hat', 'gauge', 'frito-lay', 'nestle', 'nail_file', 'vaccine', 'lego_figure', 'biohazard', 'bird_bath', 'julie', 'whip', 'pole_vaulter', 'keychain', 'ping_pong_paddle', 'park_police', 'billiard_cue', 'goalkeeper', "dunkin'_donuts", 'mahjong', 'warrior', 'video_camera', 'roaster', 'orthopaedic_&_spine_center', 'lattice', 'char', 'vhs', 'safety_belt', 'cricket_ball', 'prosthetic_leg', 'teleport', 'reptile', 'cell_phone_case', 'jazz_festival', 'dune', 'parasail', 'computer_screen', 'toilet_seat', 'billiard_ball', 'paddleboard', 'groundhog', 'long_horned_animal', 'gaming_chair', 'money_clip', 'aircraft_carrier', 'marshmallows', 'corbel', 'barber_shop', 'station', 'video', 'barcelona', 'crisper', 'dog_tag', 'gypsum', 'penny', 'tugboat', 'marbles', 'sweet_potato_fries', 'birthday', 'chocolate_chip_cookies', 'crayfish', 'prunes', 'first_aid', 'distillery', 'checkered_flag', 'seagull', 'pow', 'tools', 'drill_bit', 'valve', 'ice_rink', 'vertebra', 'mosquito_net', 'bathroom_cleaner', 'catalytic_converter', 'pinata', 'sea_cucumber', 'antique', 'gravel', 'tv_show', 'hop', 'tanker', 'lottery', 'curling_stone', 'frizerie', 'wireless_network', 'armored_car', 'badminton', 'star_fruit', 'new_era_fits', 'new_era_fit', 'goji_berries', 'hood', 'granules', 'sharpening_tool', 'pixel', 'crowd', 'theatre', 'radio_city_music_hall', 'ricoh', 'dome', 'plumbing', 'columbia', 'almond_butter', 'pipe_wrench', 'vine', 'shofar', 'parasailing', 'no_turn', 'red_bull', 'train_station_platform', 'train_station_sign', 'golden_curry_sauce_mix', 'crustacean', 'gunpowder', 'charity_stall', 'caution_sign', 'alp', 'mining_shovel', 'bar_chart', 'bucket_truck', 'bathroom_fixture', 'rickshaw', 'laundry_bag', 'love', 'deposit', 'pears', 'nectar', 'charging_station', 'oriental', 'hummingbird_feeder', 'christmas_card', 'elf', 'claw', 'swab', 'crockery', 'tear', 'wooden_chair', 'spring', 'anthropomorphic', 'hair_conditioner', 'jelly', 'organic', 'masonic_symbol', 'rhubarb', 'slippery', 'mineral', 'pharmacy', 'cashew_nut', 'coca_cola_machine', 'gadget', 'sickle', 'radio_telescope', 'army_uniform', 'hiking_trail', 'pencil_case', 'rollerblading', 'rugby', 'motorola', 'medical_equipment', 'ground', 'ankle', 'star_wars', 'pincette', 'paperclip', 'medication', 'passion_fruit', 'grilled_bread', 'bubbles', 'jollibee', 'laundry_basket', 'pearl', 'buttercream', 'iv', 'armored_truck', 'period', 'rainforest', 'band_logo', 'diet_soda', 'elmo', 'health', 'theater', 'krispy_kreme_donut', 'camper', 'pavement', 'toilet_flusher', 'minecraft', 'agrodome', 'golfers', 'rebar', 'boxer_glove', 'chickpeas', 'bath_tissue', 'network_jack', 'bible', 'earth', 'matryoshka', '$24', 'pet_carrier', 'zip_lock_bag', 'cuff', 'ice_cream_truck', 'soda_bottle', 'pita', 'fungus', 'blowtorch', 'out_of_order', 'northbound', 'wooden_bench', 'vittel', 'dvd', 'tanks', '7up', 'vga', 'drums', 'noodle', 'troll', 'towel_ring', 'litter_box', 'chickens', 'chick', 'lychee', 'man_with_stick', 'root_beer_float', 'metronome', 'hanging_fabric', 'clips', 'agenda', 'stain', 'coffee_mill', 'alliplay', 'wooden_gate', 'slice', 'kfc', 'superman', 'mint', 'electric_kettle', 'web_page', 'multimeter', 'potato_chips', 'body_lotion', 'lump', 'yurt', 'construction_crane', 'tissue_box', 'school_zone', 'patch', 'nail_polish', 'plow', 'guitar_case', 'coffee_cup_lid', 'baton', 'polo', 'holly', 'baseball_umpire', 'rust', 'magazines', 'overalls', 'cardboard_box', 'snack', 'reeds', 'smile', 'grant', 'priest', 'spire', 'nudibranch', 'pringles', 'soundbar', 'judge', 'waitress', 'casino', 'lockers', 'wakeboard', 'boxing', 'gas_station_pump', 'dog_bowl', 'egyptian_sphinx', 'nutella', 'video_atm', 'moccasin', 'colgate_toothpaste', 'pacman', 'sports_equipment', 'restaurant_menu', 'hard_hat', 'turntable', 'breadstick', 'pilot', 'melted_chocolate', 'euro_symbol', 'tanning_bed', 'pizza_cart', 'glue', 'macys', 'nintendo', 'alpaca', 'metro', 'race_number', 'frosting', 'yoyo', 'cuisinart', 'engagement_ring', 'eye_dropper', 'anime_character', 'painting_services', 'gumball_machine', 'eyeshadow', 'like_button', 'chicken_broth', 'paper_towel_roll', 'weather_station', 'mantis', 'android', 'wooden_pallet', 'basil', 'seeds', 'chopped_cheese', 'concert', 'cornucopia', 'sprinkler', 'tnt', 'scone', '4-way_stop', 'card_holder', 'hacksaw', 'leggings', 'toilet_paper_dispenser', 'mittens', 'offer', 'hero', 'electric_pole', 'pedestrian_crossing_button', 'barbados', 'tartas', 'spaghetti_sauce', 'makeup_brush', 'american_express', 'macbook_pro', 'snoopy', 'metrocard', 'candy_dispenser', 'ice_cube_tray', 'vegetarian_sauce', 'ipad', 'jukebox', 'pelvis', 'stem', 'tailor', 'stained_glass', 'shirtless', 'laboratory', 'living_room', 'fire_route', 'donut_shop', 'plastic_cup', 'corn_chips', 'air_purifier', 'barcode_scanner', 'spark_plug', 'butterball', 'poem', 'armored_figure', 'roll_of_toilet_paper', 'dunlop', 'painting_easel', 'nun', 'globus', 'monopoly', 'japanese_text', 'pedestrian_sign', 'antlers', 'revolver', 'minaret', 'contact_lens_case', 'bullet_hole', 'price', 'auto_rickshaw', 'spinning_top', 'scuba_diver', 'coals', 'game_menu', 'hill', 'electricity_pole', 'record_label', 'zipline', 'light_emitting_diode', 'croquette', 'mile_marker', 'cicada', 'red', 'protozoan', 'switch', 'orangutan', 'shelves', 'laptop_stand', 'extintor', 'luna', 'nivea', 'bandeira', 'hours', 'brussel_sprouts', 'fairgrounds', 'apple_strudel', 'carving', 'air_duct', 'baby_walker', 'airpod', 'cheetos', 'vineyard', 'oil_can', 'ironing-board', 'pac-man', 'dog_bone', 'prescott', 'evian', 'palettes', 'droplet', 'mobs', 'ceramic', 'krispy_kreme_doughnut', 'shoe_horn', 'marmite', 'mouse_trap', 'newspaper_stand', 'cookie_monster', 'espn_logo', 'world_map', 'film_strip', 'toga', 'nappy', 'automat', 'pots', 'pine', 'racecar', 'halo', 'sleigh', 'camera_strap', 'phone_case', 'playground', 'abstract_art', 'toucan', 'curtain_rod', 'wheels', 'lacrosse_racket', 'petmate', 'pcb', 'baby_crib', 'cinnamon_raisin', 'kimchi', 'utensils', 'plastic_spoon', 'plastic_wrap', 'rotary_lift', 'police_hat', 'toray', 'chaise_lounge', 'acme', 'printer_ribbon', 'tadpole', 'cyclone', 'figure', 'essential_oil', 'sunbed', 'super_bowl', 'imac', 'oil_filter', 'cog', 'charcoal_grill', 'dragon_boat', 'creeper', 'extinguisher', 'tarantula', 'golf_course', 'beacon', 'smoker', 'cordeau', 'dreamcatcher', 'farmers_market', 'camera_tripod', 'windows_logo', 'hangers', 'cotton_swabs', 'gondola', 'hurdle', 'film_clapper', 'guitar_pick', 'waffle_maker', 'nebulizer', 'torture', 'coffee_press', 'pickles', 'caviar', 'lifeguard_tower', 'curb', 'toe', 'inflatable_arch', 'spider-man', 'gas', 'windsurfer', 'buttocks', 'staff', 'hornbill', 'test', 'construction_helmet', 'film_slate', 'software', 'biscuits', 'sea_lion', 'chalk', 'cool_box', 'atp', 'garden_gnome', 'briefs', 'skate_park', 'horseshoe_crab', 'old_bay_seasoning', 'soap_bubble', 'chocolate_fountain', 'no_vehicle', 'thorn', 'baking_powder', 'earbud', 'motorcycle_shop', 'church_spire', 'openstack', 'corona', 'swirl', 'bride', 'peace_symbol', 'cake_server', 'salt_and_pepper_shaker', 'chinese_character', 'falling_rock', 'tennis_racket_bag', 'penne', 'egg_carton', 'port', 'arcade', '4-way_intersection', 'yellow', 'hippo', 'no_standing', 'hanging_bed', 'molehill', 'clover', 'wilson', 'modem', 'toner_cartridge', 'trooper', 'clone_trooper', 'painting_brush', 'tuck_box', 'wishbone', 'doorbell', 'new', 'sony_ericsson', 'catch', 'post_office', 'movie_ticket', 'bank_truck', 'pedestrian_bridge', 'antiseptic', 'fortune_cookie', 'freeway', 'christ_the_redeemer', 'inspector', 'electricity_pylon', 'taste', 'baby_carriage', 'lifeguard_chair', 'alcohol-free_ginger_syrup', 'track_light', 'railway_crossing', 'xml', 'sewing_kit', 'neck_pillow', '4-way', 'nuclear', 'inkwell', 'stamp', 'botox', 'gynecological_speculum', 'bathroom_countertop', 'caricature', 'l', 'chocolate_chip_cookie', 'resort', 'edamame', 'fish_hook', 'stuffed_toy', 'house_number', 'multitool', 'asbestos', 'oral-b', 'fairy_staff', 'hospital_gown', 'artillery', 'scrabble', 'yield_to_pedestrians', 'blackcurrant', 'project_management', 'microsd_card', 'caramel', 'marijuana', 'one_way_sign', 'power_bank', 'bow_and_arrow', 'black', 'ant_man', 'film_clapperboard', 'evergreen_park', 'cemetery', 'hot_chocolate', 'hobo_inn', 'christmas_santa_claus', 'pawnshop', 'caravan', 'beaded', 'bbq', 'volunteer', 'walker', 'font', 'high_voltage', 'window_shutter', 'ticket_vending_machine', 'partition', 'newstalk', 'general', 'lic', 'ninja', 'tall_object', 'laser_light', 'travel_center', 'facebook', 'turkey_stuffing', 'chocolate_chip', 'striped_awning', 'first_aid_box', 'protein_bar', 'sauna', 'hula_skirt', 'mop_bucket', 'ramekin', 'ski_jacket', 'card_case', 'headlamp', 'billiard_hall', 'hermit_crab', 'funnel_cake', 'no_vehicles', 'seafood', 'cough_medicine', 'champignon', 'fishbone', 'slipper', 'pier', 'longhorn', 'seats', 'sports_ball', 'king', 'birthday_party', 'schoolgirl', 'skillet', 'free_press', 'washington_nationals_logo', 'copyright_symbol', 'wedding_couple', 'tie_clip', 'budget', 'stop', 'dried_fruit', 'office_furniture', 'landmark', 'dodecahedron', 'allplay', 'smoke_stack', 'cookie_cutter', 'dill', 'zodiac', 'dental_implant', 'school_speed_limit', 'transport', 'olympic_center', 'fritter', 'tourist_information', 'angry_bird', 'bow_tie', 'no_stopping_or_changing', 'chard', 'infinity_symbol', 'parthenon', 'party', 'cooking_book', 'motor_oil', 'termite', 'orchard', 'brake_pads', 'vanilla', 'hiker', 'stopwatch', 'led', 'burner', 'fly_fishing_hook', 'tool_bag', 'iguana', 'handcuffs', 'gates', 'louvre', 'krishna', 'fish_food', 'bowel', 'baseball_shirt', 'spring_roll', 'flow', 'iphone', 'chlordiazepoxide', 'anthill', 'bobblehead', 'web', 'crochet', 'laser_ranger', 'logistics', 'push_button', 'ankle_brace', 'moses_basket', 'ad', 'mints', 'hand_cream', 'x-ray_machine', 'non-fried', 'air_compressor', 'security', 'bakeries', 'pork', 'laptop_bag', 'cooking_pot', 'caduceus', 'tuberculosis', 'web_banner', 'chess', 'antiseptic_oral_rinse', 'chalkboard', 'sunscreen', 'no_sign', 'hanging', 'gingerbread', 'newspaper_vending_machine', 'bazaar', "bird's_nest", 'power_distribution_box', 'fingertip', 'bladder', 'room', 'grandfather_clock', 'sling', 'internet_cafe', 'records', 'pool_cue', 'puddle', 'n/a', 'bunch_of_herbs', 'skin_cleanser', 'press', 'phosphate', 'l-shaped_object', 'fingers', 'eyeball', 'lions', 'dial', 'scope', 'frost', 'gobble', 'wood_recorder', 'toothpaste_dispenser', 'bake', 'pizza_shovel', 'yoga_mat', 'shoe_repair', 'gin', 'meatballs', 'komodo_dragon', 'enel', 'sheriff', 'metal_detector', 'no_left_turn', 'wooden_board', 'fisherman', 'stamps', 'fragment', 'vegetable_oil', 'teardrop', 'dryer', 'wooden_bowl', 'sound_system', 'humidifier', 'lotus_root', 'running_room', 'moustache', 'dentures', 'batteries', 'psychic', 'algae', 'motorcycle_suit', 'oreo', 'checkers', 'doughnut_shop', 'centrifuge', 'pine_needle', 'grilled_meat', 'news_anchor', 'christmas_gift', 'millipede', 'lacrosse_stick', 'steamer', 'copter', 'highland_spring', 'paint_spray_gun', 'cubes', 'roller_skater', 'hand_saw', 'band-aid', 'police_station', 'market', 'sprouts', 'retro', 'cattle', 'star_anise', 'silk', 'cola', 'warthog', 'toll_booth', 'logs', 'dates', 'm&m', 'wigs', 'doily', 'orthopaedic', 'chicago', 'ticket_machine', 'webpage', 'curved_path', 'handgun', 'fishing_hook', 'grinch', 'hay_bag', 'discount_tag', 'brief', 'comcast', 'lorry', 'birdseed', 'bunch_of_leaves', 'alarm', 'highlighter', 'vegetable_garden', 'statue_of_liberty', 'spider_web', 'television_tower', 'flip_chart', 'reservation', 'bob_marley', 'snow_plow', 'silver', 'sausage_roll', 'washing_detergent', 'hair_clipper', 'fila', 'ice_axe', 'at&t', 'lumber', 'carbon_dioxide', 'bbq_sauce', 'mussel', 'paper_towel_dispenser', 'mickey_mouse', 'department_of_agriculture', 'utility_work_ahead', 'saturn', 'hammerhead', 'clam', 'wild_boar', 'roman_soldier', 'engine_part', 'matcha_whisk', 'mezuzah', 'worker', 'cosmetic', 'inflatable_balloon', 'welder', "pimm's_no.1", 'armored_person', 'file_folder', 'food_container', 'bath_product', 'dancing', 'bath_bomb', 'ladder_truck', 'speed_hump', 'planter', 'olympus', 'birds', 'dentist_tool', 'vines', 'cradle', 'rainbow_flag', 'olympic_symbol', 'pulp', 'guacamole', 'military', 'sweat_spoon', 'ventilator', 'sea_otter', 'turnbuckle', 'smurf', 'ouija', 'nail_clipper', 'cbn', 'bon_voyage', 'jewellery', 'pile', 'longan', 'horse_halter', 'phonograph', 'go_board', 'bedroom', 'grapple', 'speed_bump', 'ammonite', 'i-beam', 'xbox', 'tailoring', 'spicy_sausage', 'starfruit', 'baseball_card', 'bike_lane', 'carport', 'electric', 'checkbox', 'inflatable_chair', 'qr-code', 'cypress', 'manicure_set', 'email', 'ice_cream_scoop', 'racer', 'adidas_bag', 'hash_browns', 'dvds', 'gator', 'baseball_pitcher', 'aquinas_college', 'l-shaped', 'duster', 'mite', 'shoulder', 'outlet', 'bento_box', 'turret', 'ride', 'no_smoking', 'burlap', 'wooden_carving', 'thai_cuisine', 'neem', 'armillary_sphere', 'ping_pong_ball', 'coat_rack', 'sports_grill', 'smoking_paper', 'magnifying_glass', 'raisin', '螺丝', 'cryptocurrency', 'breakfast_cereal', 'cardinal', 'lithium_battery', 'merger', 'soil', 'pressure_gauge', 'aoki', 'school_bus_stop', 'polystyrene', 'dent', 'air_vent', 'emergency_phone', 'inflatable_animal', 'corn_chip', 'actavis', 'dunkin_donuts', 'gasometer', 'cogwheel', 'fibre', 'jade', 'hanging_gallows', 'ice_scraper', 'rails', 'electric_outlet', 'dishes', 'lafleur_hair_&_nail_spa', 'softball_player', 'archer', 'foreclosure', 'object', 'forum', 'gas_cylinder', 'yin_yang', 'stack_of_boxes', 'hot_dough', 'matzo', 'trestle', 'stack_of_blocks', 'dj', 'sugar_spoon', 'potty', 'burning_candle', 'shuttle', 'baggage_belt', 'bassoon', 'staples', 'maze', 'groom', 'wishing_well', 'dog_food', 'french_horn', 'macaroon', 'hole_punch', 'floral', 'drug_store', 'drug', 'link_box', 'cleaning', 'sewer', 'jackhammer', 'bus_stop_sign', 'clothes_dryer', 'billabong', "judge's_gavel", 'drape', 'collagen', 'bannister', 'wires', 'medel', 'lawn_mower_blade', 'valero', 'olympic_flag', 'sleeping_pad', 'apple_tv', 'steam_engine', 'photocopier', 'base', 'recycling_symbol', 'papertowel', 'taxi_stand', 'italian_sausage', 'macaroni_and_cheese', 'ink_cartridge', 'beignet', 'shaving_machine', 'roman', 'beer_coaster', 'aloe_vera', 'lady', 'chocolate_hazelnut', 'vise', 'stamping_tool', 'bungee', 'nassfeld', 'taco_seasoning', 'trays', 'camouflage', 'cowboy_boot', 'coke', 'keyboard_key', 'fishing_pole', 'kettlebell', 'ups_package', 'dry_chemical_extinguisher', 'flying_discs_association', 'makeup', 'tennis_net', 'sloth', 'electricity_box', 'stack_of_wood', 'isle_of_arran', 'cones', "jack_daniel's_bottle", 'cancer_research_foundation', 'bus_bench', 'garnier', 'blood_pressure_gauge', 'beetroot', 'rolls', 'sundial', 'possum', 'lunch_box', 'scrummy', 'marinara_sauce', 'baking_soda', 'cold_drinks', 'kitchenaid', 'tyres', 'peep', 'door_hanger', 'gali', 'ticket_scanner', 'florida', 'ginkgo', '3d_model', 'flask', 'photograph', 'cookie_jar', 'lithograph', 'fire_lane', 'speed_table', 'cd_drive', 'persian', 'bells', 'pin_cushion', "mcalister's_deli", 'leonardo_da_vinci', 'paci', 'smiley', 'jigsaw_puzzle_piece', 'smoking_sign', 'military_uniform', 'sumo', 'iv_pole', 'music', 'geography', 'neighborhood_watch', 'tillamook', 'swastika', 'pie_chart', 'cereal_bar', 'wooden_handle', 'frankfurter', 'card_reader', 'zine', 'fruit_punch', 'papyrus', 'soccer_shirt', 'door_lock', 'brussels_sprout', 'motocross_rider', 'cauli', 'litter', 'pogo_stick', 'cameraman', 'bill_shrink', 'hinges', 'festival', 'broth', 'dough', 'cupcake_mold', 'lithium-ion_battery', 'ultraman', 'heineken', 'teletubby', 'bus_lane', 'hotspot'),
        'palette':
        None
    }

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        try:
            import projects.cotpl.reprod.lvis as lvis
            if getattr(lvis, '__version__', '0') >= '10.5.3':
                warnings.warn(
                    'mmlvis is deprecated, please install official lvis-api by "pip install git+https://github.com/lvis-dataset/lvis-api.git"',  # noqa: E501
                    UserWarning)
            from projects.cotpl.reprod.lvis import LVIS
        except ImportError:
            raise ImportError(
                'Package lvis is not installed. Please run "pip install git+https://github.com/lvis-dataset/lvis-api.git".'  # noqa: E501
            )
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.lvis = LVIS(local_path)
        self.cat_ids = self.lvis.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.lvis.cat_img_map)

        img_ids = self.lvis.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.lvis.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id
            if raw_img_info['file_name'].startswith('COCO'):
                # Convert form the COCO 2014 file naming convention of
                # COCO_[train/val/test]2014_000000000000.jpg to the 2017
                # naming convention of 000000000000.jpg
                # (LVIS v1 will fix this naming issue)
                raw_img_info['file_name'] = raw_img_info['file_name'][-16:]
            ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.lvis.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.lvis

        return data_list


LVISDataset = LVISV05Dataset
DATASETS.register_module(name='LVISDataset', module=LVISDataset)


@DATASETS.register_module()
class LVISV1Dataset(LVISDataset):
    """LVIS v1 dataset for detection."""

    METAINFO = {
        'classes':
        ('aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock',
         'alcohol', 'alligator', 'almond', 'ambulance', 'amplifier', 'anklet',
         'antenna', 'apple', 'applesauce', 'apricot', 'apron', 'aquarium',
         'arctic_(type_of_shoe)', 'armband', 'armchair', 'armoire', 'armor',
         'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer',
         'avocado', 'award', 'awning', 'ax', 'baboon', 'baby_buggy',
         'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel',
         'bagpipe', 'baguet', 'bait', 'ball', 'ballet_skirt', 'balloon',
         'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna', 'banjo',
         'banner', 'barbell', 'barge', 'barrel', 'barrette', 'barrow',
         'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap',
         'baseball_glove', 'basket', 'basketball', 'bass_horn', 'bat_(animal)',
         'bath_mat', 'bath_towel', 'bathrobe', 'bathtub', 'batter_(food)',
         'battery', 'beachball', 'bead', 'bean_curd', 'beanbag', 'beanie',
         'bear', 'bed', 'bedpan', 'bedspread', 'cow', 'beef_(food)', 'beeper',
         'beer_bottle', 'beer_can', 'beetle', 'bell', 'bell_pepper', 'belt',
         'belt_buckle', 'bench', 'beret', 'bib', 'Bible', 'bicycle', 'visor',
         'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath',
         'birdcage', 'birdhouse', 'birthday_cake', 'birthday_card',
         'pirate_flag', 'black_sheep', 'blackberry', 'blackboard', 'blanket',
         'blazer', 'blender', 'blimp', 'blinker', 'blouse', 'blueberry',
         'gameboard', 'boat', 'bob', 'bobbin', 'bobby_pin', 'boiled_egg',
         'bolo_tie', 'deadbolt', 'bolt', 'bonnet', 'book', 'bookcase',
         'booklet', 'bookmark', 'boom_microphone', 'boot', 'bottle',
         'bottle_opener', 'bouquet', 'bow_(weapon)',
         'bow_(decorative_ribbons)', 'bow-tie', 'bowl', 'pipe_bowl',
         'bowler_hat', 'bowling_ball', 'box', 'boxing_glove', 'suspenders',
         'bracelet', 'brass_plaque', 'brassiere', 'bread-bin', 'bread',
         'breechcloth', 'bridal_gown', 'briefcase', 'broccoli', 'broach',
         'broom', 'brownie', 'brussels_sprouts', 'bubble_gum', 'bucket',
         'horse_buggy', 'bull', 'bulldog', 'bulldozer', 'bullet_train',
         'bulletin_board', 'bulletproof_vest', 'bullhorn', 'bun', 'bunk_bed',
         'buoy', 'burrito', 'bus_(vehicle)', 'business_card', 'butter',
         'butterfly', 'button', 'cab_(taxi)', 'cabana', 'cabin_car', 'cabinet',
         'locker', 'cake', 'calculator', 'calendar', 'calf', 'camcorder',
         'camel', 'camera', 'camera_lens', 'camper_(vehicle)', 'can',
         'can_opener', 'candle', 'candle_holder', 'candy_bar', 'candy_cane',
         'walking_cane', 'canister', 'canoe', 'cantaloup', 'canteen',
         'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino',
         'car_(automobile)', 'railcar_(part_of_a_train)', 'elevator_car',
         'car_battery', 'identity_card', 'card', 'cardigan', 'cargo_ship',
         'carnation', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton',
         'cash_register', 'casserole', 'cassette', 'cast', 'cat',
         'cauliflower', 'cayenne_(spice)', 'CD_player', 'celery',
         'cellular_telephone', 'chain_mail', 'chair', 'chaise_longue',
         'chalice', 'chandelier', 'chap', 'checkbook', 'checkerboard',
         'cherry', 'chessboard', 'chicken_(animal)', 'chickpea',
         'chili_(vegetable)', 'chime', 'chinaware', 'crisp_(potato_chip)',
         'poker_chip', 'chocolate_bar', 'chocolate_cake', 'chocolate_milk',
         'chocolate_mousse', 'choker', 'chopping_board', 'chopstick',
         'Christmas_tree', 'slide', 'cider', 'cigar_box', 'cigarette',
         'cigarette_case', 'cistern', 'clarinet', 'clasp', 'cleansing_agent',
         'cleat_(for_securing_rope)', 'clementine', 'clip', 'clipboard',
         'clippers_(for_plants)', 'cloak', 'clock', 'clock_tower',
         'clothes_hamper', 'clothespin', 'clutch_bag', 'coaster', 'coat',
         'coat_hanger', 'coatrack', 'cock', 'cockroach', 'cocoa_(beverage)',
         'coconut', 'coffee_maker', 'coffee_table', 'coffeepot', 'coil',
         'coin', 'colander', 'coleslaw', 'coloring_material',
         'combination_lock', 'pacifier', 'comic_book', 'compass',
         'computer_keyboard', 'condiment', 'cone', 'control',
         'convertible_(automobile)', 'sofa_bed', 'cooker', 'cookie',
         'cooking_utensil', 'cooler_(for_food)', 'cork_(bottle_plug)',
         'corkboard', 'corkscrew', 'edible_corn', 'cornbread', 'cornet',
         'cornice', 'cornmeal', 'corset', 'costume', 'cougar', 'coverall',
         'cowbell', 'cowboy_hat', 'crab_(animal)', 'crabmeat', 'cracker',
         'crape', 'crate', 'crayon', 'cream_pitcher', 'crescent_roll', 'crib',
         'crock_pot', 'crossbar', 'crouton', 'crow', 'crowbar', 'crown',
         'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch',
         'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup',
         'cupboard', 'cupcake', 'hair_curler', 'curling_iron', 'curtain',
         'cushion', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'dartboard',
         'date_(fruit)', 'deck_chair', 'deer', 'dental_floss', 'desk',
         'detergent', 'diaper', 'diary', 'die', 'dinghy', 'dining_table',
         'tux', 'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher',
         'dishwasher_detergent', 'dispenser', 'diving_board', 'Dixie_cup',
         'dog', 'dog_collar', 'doll', 'dollar', 'dollhouse', 'dolphin',
         'domestic_ass', 'doorknob', 'doormat', 'doughnut', 'dove',
         'dragonfly', 'drawer', 'underdrawers', 'dress', 'dress_hat',
         'dress_suit', 'dresser', 'drill', 'drone', 'dropper',
         'drum_(musical_instrument)', 'drumstick', 'duck', 'duckling',
         'duct_tape', 'duffel_bag', 'dumbbell', 'dumpster', 'dustpan', 'eagle',
         'earphone', 'earplug', 'earring', 'easel', 'eclair', 'eel', 'egg',
         'egg_roll', 'egg_yolk', 'eggbeater', 'eggplant', 'electric_chair',
         'refrigerator', 'elephant', 'elk', 'envelope', 'eraser', 'escargot',
         'eyepatch', 'falcon', 'fan', 'faucet', 'fedora', 'ferret',
         'Ferris_wheel', 'ferry', 'fig_(fruit)', 'fighter_jet', 'figurine',
         'file_cabinet', 'file_(tool)', 'fire_alarm', 'fire_engine',
         'fire_extinguisher', 'fire_hose', 'fireplace', 'fireplug',
         'first-aid_kit', 'fish', 'fish_(food)', 'fishbowl', 'fishing_rod',
         'flag', 'flagpole', 'flamingo', 'flannel', 'flap', 'flash',
         'flashlight', 'fleece', 'flip-flop_(sandal)', 'flipper_(footwear)',
         'flower_arrangement', 'flute_glass', 'foal', 'folding_chair',
         'food_processor', 'football_(American)', 'football_helmet',
         'footstool', 'fork', 'forklift', 'freight_car', 'French_toast',
         'freshener', 'frisbee', 'frog', 'fruit_juice', 'frying_pan', 'fudge',
         'funnel', 'futon', 'gag', 'garbage', 'garbage_truck', 'garden_hose',
         'gargle', 'gargoyle', 'garlic', 'gasmask', 'gazelle', 'gelatin',
         'gemstone', 'generator', 'giant_panda', 'gift_wrap', 'ginger',
         'giraffe', 'cincture', 'glass_(drink_container)', 'globe', 'glove',
         'goat', 'goggles', 'goldfish', 'golf_club', 'golfcart',
         'gondola_(boat)', 'goose', 'gorilla', 'gourd', 'grape', 'grater',
         'gravestone', 'gravy_boat', 'green_bean', 'green_onion', 'griddle',
         'grill', 'grits', 'grizzly', 'grocery_bag', 'guitar', 'gull', 'gun',
         'hairbrush', 'hairnet', 'hairpin', 'halter_top', 'ham', 'hamburger',
         'hammer', 'hammock', 'hamper', 'hamster', 'hair_dryer', 'hand_glass',
         'hand_towel', 'handcart', 'handcuff', 'handkerchief', 'handle',
         'handsaw', 'hardback_book', 'harmonium', 'hat', 'hatbox', 'veil',
         'headband', 'headboard', 'headlight', 'headscarf', 'headset',
         'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet',
         'heron', 'highchair', 'hinge', 'hippopotamus', 'hockey_stick', 'hog',
         'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'hookah',
         'hornet', 'horse', 'hose', 'hot-air_balloon', 'hotplate', 'hot_sauce',
         'hourglass', 'houseboat', 'hummingbird', 'hummus', 'polar_bear',
         'icecream', 'popsicle', 'ice_maker', 'ice_pack', 'ice_skate',
         'igniter', 'inhaler', 'iPod', 'iron_(for_clothing)', 'ironing_board',
         'jacket', 'jam', 'jar', 'jean', 'jeep', 'jelly_bean', 'jersey',
         'jet_plane', 'jewel', 'jewelry', 'joystick', 'jumpsuit', 'kayak',
         'keg', 'kennel', 'kettle', 'key', 'keycard', 'kilt', 'kimono',
         'kitchen_sink', 'kitchen_table', 'kite', 'kitten', 'kiwi_fruit',
         'knee_pad', 'knife', 'knitting_needle', 'knob', 'knocker_(on_a_door)',
         'koala', 'lab_coat', 'ladder', 'ladle', 'ladybug', 'lamb_(animal)',
         'lamb-chop', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard',
         'laptop_computer', 'lasagna', 'latch', 'lawn_mower', 'leather',
         'legging_(clothing)', 'Lego', 'legume', 'lemon', 'lemonade',
         'lettuce', 'license_plate', 'life_buoy', 'life_jacket', 'lightbulb',
         'lightning_rod', 'lime', 'limousine', 'lion', 'lip_balm', 'liquor',
         'lizard', 'log', 'lollipop', 'speaker_(stereo_equipment)', 'loveseat',
         'machine_gun', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)',
         'mallard', 'mallet', 'mammoth', 'manatee', 'mandarin_orange',
         'manger', 'manhole', 'map', 'marker', 'martini', 'mascot',
         'mashed_potato', 'masher', 'mask', 'mast', 'mat_(gym_equipment)',
         'matchbox', 'mattress', 'measuring_cup', 'measuring_stick',
         'meatball', 'medicine', 'melon', 'microphone', 'microscope',
         'microwave_oven', 'milestone', 'milk', 'milk_can', 'milkshake',
         'minivan', 'mint_candy', 'mirror', 'mitten', 'mixer_(kitchen_tool)',
         'money', 'monitor_(computer_equipment) computer_monitor', 'monkey',
         'motor', 'motor_scooter', 'motor_vehicle', 'motorcycle',
         'mound_(baseball)', 'mouse_(computer_equipment)', 'mousepad',
         'muffin', 'mug', 'mushroom', 'music_stool', 'musical_instrument',
         'nailfile', 'napkin', 'neckerchief', 'necklace', 'necktie', 'needle',
         'nest', 'newspaper', 'newsstand', 'nightshirt',
         'nosebag_(for_animals)', 'noseband_(for_animals)', 'notebook',
         'notepad', 'nut', 'nutcracker', 'oar', 'octopus_(food)',
         'octopus_(animal)', 'oil_lamp', 'olive_oil', 'omelet', 'onion',
         'orange_(fruit)', 'orange_juice', 'ostrich', 'ottoman', 'oven',
         'overalls_(clothing)', 'owl', 'packet', 'inkpad', 'pad', 'paddle',
         'padlock', 'paintbrush', 'painting', 'pajamas', 'palette',
         'pan_(for_cooking)', 'pan_(metal_container)', 'pancake', 'pantyhose',
         'papaya', 'paper_plate', 'paper_towel', 'paperback_book',
         'paperweight', 'parachute', 'parakeet', 'parasail_(sports)',
         'parasol', 'parchment', 'parka', 'parking_meter', 'parrot',
         'passenger_car_(part_of_a_train)', 'passenger_ship', 'passport',
         'pastry', 'patty_(food)', 'pea_(food)', 'peach', 'peanut_butter',
         'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'wooden_leg',
         'pegboard', 'pelican', 'pen', 'pencil', 'pencil_box',
         'pencil_sharpener', 'pendulum', 'penguin', 'pennant', 'penny_(coin)',
         'pepper', 'pepper_mill', 'perfume', 'persimmon', 'person', 'pet',
         'pew_(church_bench)', 'phonebook', 'phonograph_record', 'piano',
         'pickle', 'pickup_truck', 'pie', 'pigeon', 'piggy_bank', 'pillow',
         'pin_(non_jewelry)', 'pineapple', 'pinecone', 'ping-pong_ball',
         'pinwheel', 'tobacco_pipe', 'pipe', 'pistol', 'pita_(bread)',
         'pitcher_(vessel_for_liquid)', 'pitchfork', 'pizza', 'place_mat',
         'plate', 'platter', 'playpen', 'pliers', 'plow_(farm_equipment)',
         'plume', 'pocket_watch', 'pocketknife', 'poker_(fire_stirring_tool)',
         'pole', 'polo_shirt', 'poncho', 'pony', 'pool_table', 'pop_(soda)',
         'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot',
         'potato', 'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn',
         'pretzel', 'printer', 'projectile_(weapon)', 'projector', 'propeller',
         'prune', 'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'pumpkin',
         'puncher', 'puppet', 'puppy', 'quesadilla', 'quiche', 'quilt',
         'rabbit', 'race_car', 'racket', 'radar', 'radiator', 'radio_receiver',
         'radish', 'raft', 'rag_doll', 'raincoat', 'ram_(animal)', 'raspberry',
         'rat', 'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt',
         'recliner', 'record_player', 'reflector', 'remote_control',
         'rhinoceros', 'rib_(food)', 'rifle', 'ring', 'river_boat', 'road_map',
         'robe', 'rocking_chair', 'rodent', 'roller_skate', 'Rollerblade',
         'rolling_pin', 'root_beer', 'router_(computer_equipment)',
         'rubber_band', 'runner_(carpet)', 'plastic_bag',
         'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'safety_pin',
         'sail', 'salad', 'salad_plate', 'salami', 'salmon_(fish)',
         'salmon_(food)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)',
         'sandwich', 'satchel', 'saucepan', 'saucer', 'sausage', 'sawhorse',
         'saxophone', 'scale_(measuring_instrument)', 'scarecrow', 'scarf',
         'school_bus', 'scissors', 'scoreboard', 'scraper', 'screwdriver',
         'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seaplane',
         'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark',
         'sharpener', 'Sharpie', 'shaver_(electric)', 'shaving_cream', 'shawl',
         'shears', 'sheep', 'shepherd_dog', 'sherbert', 'shield', 'shirt',
         'shoe', 'shopping_bag', 'shopping_cart', 'short_pants', 'shot_glass',
         'shoulder_bag', 'shovel', 'shower_head', 'shower_cap',
         'shower_curtain', 'shredder_(for_paper)', 'signboard', 'silo', 'sink',
         'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole',
         'skirt', 'skullcap', 'sled', 'sleeping_bag', 'sling_(bandage)',
         'slipper_(footwear)', 'smoothie', 'snake', 'snowboard', 'snowman',
         'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa', 'softball',
         'solar_array', 'sombrero', 'soup', 'soup_bowl', 'soupspoon',
         'sour_cream', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)',
         'spatula', 'spear', 'spectacles', 'spice_rack', 'spider', 'crawfish',
         'sponge', 'spoon', 'sportswear', 'spotlight', 'squid_(food)',
         'squirrel', 'stagecoach', 'stapler_(stapling_machine)', 'starfish',
         'statue_(sculpture)', 'steak_(food)', 'steak_knife', 'steering_wheel',
         'stepladder', 'step_stool', 'stereo_(sound_system)', 'stew',
         'stirrer', 'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove',
         'strainer', 'strap', 'straw_(for_drinking)', 'strawberry',
         'street_sign', 'streetlight', 'string_cheese', 'stylus', 'subwoofer',
         'sugar_bowl', 'sugarcane_(plant)', 'suit_(clothing)', 'sunflower',
         'sunglasses', 'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants',
         'sweatband', 'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit',
         'sword', 'syringe', 'Tabasco_sauce', 'table-tennis_table', 'table',
         'table_lamp', 'tablecloth', 'tachometer', 'taco', 'tag', 'taillight',
         'tambourine', 'army_tank', 'tank_(storage_vessel)',
         'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure',
         'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup',
         'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth',
         'telephone_pole', 'telephoto_lens', 'television_camera',
         'television_set', 'tennis_ball', 'tennis_racket', 'tequila',
         'thermometer', 'thermos_bottle', 'thermostat', 'thimble', 'thread',
         'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer',
         'tinfoil', 'tinsel', 'tissue_paper', 'toast_(food)', 'toaster',
         'toaster_oven', 'toilet', 'toilet_tissue', 'tomato', 'tongs',
         'toolbox', 'toothbrush', 'toothpaste', 'toothpick', 'cover',
         'tortilla', 'tow_truck', 'towel', 'towel_rack', 'toy',
         'tractor_(farm_equipment)', 'traffic_light', 'dirt_bike',
         'trailer_truck', 'train_(railroad_vehicle)', 'trampoline', 'tray',
         'trench_coat', 'triangle_(musical_instrument)', 'tricycle', 'tripod',
         'trousers', 'truck', 'truffle_(chocolate)', 'trunk', 'vat', 'turban',
         'turkey_(food)', 'turnip', 'turtle', 'turtleneck_(clothing)',
         'typewriter', 'umbrella', 'underwear', 'unicycle', 'urinal', 'urn',
         'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 'vest',
         'videotape', 'vinegar', 'violin', 'vodka', 'volleyball', 'vulture',
         'waffle', 'waffle_iron', 'wagon', 'wagon_wheel', 'walking_stick',
         'wall_clock', 'wall_socket', 'wallet', 'walrus', 'wardrobe',
         'washbasin', 'automatic_washer', 'watch', 'water_bottle',
         'water_cooler', 'water_faucet', 'water_heater', 'water_jug',
         'water_gun', 'water_scooter', 'water_ski', 'water_tower',
         'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake',
         'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream',
         'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)',
         'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket',
         'wineglass', 'blinder_(for_horses)', 'wok', 'wolf', 'wooden_spoon',
         'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yogurt',
         'yoke_(animal_equipment)', 'zebra', 'zucchini'),
        'palette':
        None
    }

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        try:
            import projects.cotpl.reprod.lvis as lvis
            if getattr(lvis, '__version__', '0') >= '10.5.3':
                warnings.warn(
                    'mmlvis is deprecated, please install official lvis-api by "pip install git+https://github.com/lvis-dataset/lvis-api.git"',  # noqa: E501
                    UserWarning)
            from projects.cotpl.reprod.lvis import LVIS
        except ImportError:
            raise ImportError(
                'Package lvis is not installed. Please run "pip install git+https://github.com/lvis-dataset/lvis-api.git".'  # noqa: E501
            )
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.lvis = LVIS(local_path)
        self.cat_ids = self.lvis.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.lvis.cat_img_map)

        img_ids = self.lvis.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.lvis.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id
            # coco_url is used in LVISv1 instead of file_name
            # e.g. http://images.cocodataset.org/train2017/000000391895.jpg
            # train/val split in specified in url
            raw_img_info['file_name'] = raw_img_info['coco_url'].replace(
                'http://images.cocodataset.org/', '')
            ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.lvis.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)
            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.lvis

        return data_list
