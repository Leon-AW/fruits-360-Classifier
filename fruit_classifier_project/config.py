# --- Configuration ---

# --- Model and Image Settings ---
MODEL_PATH = "fruit_classifier_mobilenet_best_v2.keras"
IMG_WIDTH = 100  # Should match the model's expected input
IMG_HEIGHT = 100 # Should match the model's expected input

# --- UI Display Settings ---
TOP_N_PREDICTIONS_DISPLAY = 5 # For correction dialog
MAX_SEARCH_RESULTS_DISPLAY = 10 # For correction dialog search
DEFAULT_WINDOW_GEOMETRY = "1000x700"
CART_THUMBNAIL_SIZE = (55, 55)
CART_ROW_HEIGHT = 60

# --- Directories ---
USER_DATA_SUBDIR = "user_corrected_data" # Subdirectory name for user corrected data

# --- Class Names (Raw Data) ---
# User provided a 1-indexed list with format "1: Apple 10"
CLASS_NAMES_RAW = """
1: Apple 10
2: Apple 11
3: Apple 12
4: Apple 13
5: Apple 14
6: Apple 17
7: Apple 18
8: Apple 19
9: Apple 5
10: Apple 6
11: Apple 7
12: Apple 8
13: Apple 9
14: Apple Braeburn 1
15: Apple Core 1
16: Apple Crimson Snow 1
17: Apple Golden 1
18: Apple Golden 2
19: Apple Golden 3
20: Apple Granny Smith 1
21: Apple Pink Lady 1
22: Apple Red 1
23: Apple Red 2
24: Apple Red 3
25: Apple Red Delicious 1
26: Apple Red Yellow 1
27: Apple Red Yellow 2
28: Apple Rotten 1
29: Apple hit 1
30: Apple worm 1
31: Apricot 1
32: Avocado 1
33: Avocado Black 1
34: Avocado Green 1
35: Avocado ripe 1
36: Banana 1
37: Banana 3
38: Banana 4
39: Banana Lady Finger 1
40: Banana Red 1
41: Beans 1
42: Beetroot 1
43: Blackberrie 1
44: Blackberrie 2
45: Blackberrie half rippen 1
46: Blackberrie not rippen 1
47: Blueberry 1
48: Cabbage red 1
49: Cabbage white 1
50: Cactus fruit 1
51: Cactus fruit green 1
52: Cactus fruit red 1
53: Caju seed 1
54: Cantaloupe 1
55: Cantaloupe 2
56: Carambula 1
57: Carrot 1
58: Cauliflower 1
59: Cherimoya 1
60: Cherry 1
61: Cherry 2
62: Cherry 3
63: Cherry 4
64: Cherry 5
65: Cherry Rainier 1
66: Cherry Rainier 2
67: Cherry Rainier 3
68: Cherry Sour 1
69: Cherry Wax Black 1
70: Cherry Wax Red 1
71: Cherry Wax Red 2
72: Cherry Wax Red 3
73: Cherry Wax Yellow 1
74: Cherry Wax not ripen 1
75: Cherry Wax not ripen 2
76: Chestnut 1
77: Clementine 1
78: Cocos 1
79: Corn 1
80: Corn Husk 1
81: Cucumber 1
82: Cucumber 10
83: Cucumber 11
84: Cucumber 3
85: Cucumber 4
86: Cucumber 5
87: Cucumber 7
88: Cucumber 9
89: Cucumber Ripe 1
90: Cucumber Ripe 2
91: Dates 1
92: Eggplant 1
93: Eggplant long 1
94: Fig 1
95: Ginger Root 1
96: Gooseberry 1
97: Granadilla 1
98: Grape Blue 1
99: Grape Pink 1
100: Grape White 1
101: Grape White 2
102: Grape White 3
103: Grape White 4
104: Grapefruit Pink 1
105: Grapefruit White 1
106: Guava 1
107: Hazelnut 1
108: Huckleberry 1
109: Kaki 1
110: Kiwi 1
111: Kohlrabi 1
112: Kumquats 1
113: Lemon 1
114: Lemon Meyer 1
115: Limes 1
116: Lychee 1
117: Mandarine 1
118: Mango 1
119: Mango Red 1
120: Mangostan 1
121: Maracuja 1
122: Melon Piel de Sapo 1
123: Mulberry 1
124: Nectarine 1
125: Nectarine Flat 1
126: Nut Forest 1
127: Nut Pecan 1
128: Onion Red 1
129: Onion Red Peeled 1
130: Onion White 1
131: Orange 1
132: Papaya 1
133: Passion Fruit 1
134: Peach 1
135: Peach 2
136: Peach Flat 1
137: Pear 1
138: Pear 2
139: Pear 3
140: Pear Abate 1
141: Pear Forelle 1
142: Pear Kaiser 1
143: Pear Monster 1
144: Pear Red 1
145: Pear Stone 1
146: Pear Williams 1
147: Pepino 1
148: Pepper Green 1
149: Pepper Orange 1
150: Pepper Red 1
151: Pepper Yellow 1
152: Physalis 1
153: Physalis with Husk 1
154: Pineapple 1
155: Pineapple Mini 1
156: Pistachio 1
157: Pitahaya Red 1
158: Plum 1
159: Plum 2
160: Plum 3
161: Pomegranate 1
162: Pomelo Sweetie 1
163: Potato Red 1
164: Potato Red Washed 1
165: Potato Sweet 1
166: Potato White 1
167: Quince 1
168: Quince 2
169: Quince 3
170: Quince 4
171: Rambutan 1
172: Raspberry 1
173: Redcurrant 1
174: Salak 1
175: Strawberry 1
176: Strawberry Wedge 1
177: Tamarillo 1
178: Tangelo 1
179: Tomato 1
180: Tomato 10
181: Tomato 2
182: Tomato 3
183: Tomato 4
184: Tomato 5
185: Tomato 7
186: Tomato 8
187: Tomato 9
188: Tomato Cherry Maroon 1
189: Tomato Cherry Orange 1
190: Tomato Cherry Red 1
191: Tomato Cherry Red 2
192: Tomato Cherry Yellow 1
193: Tomato Heart 1
194: Tomato Maroon 1
195: Tomato Maroon 2
196: Tomato Yellow 1
197: Tomato not Ripen 1
198: Walnut 1
199: Watermelon 1
200: Zucchini 1
201: Zucchini dark 1
"""

# --- Product Data (Prices per kg, Average weight in kg) ---
PRODUCT_DATA = {
    "Apple": {"price_per_kg": 2.50, "avg_weight": 0.150, "typical_min_weight": 0.120, "typical_max_weight": 0.180},
    "Apricot": {"price_per_kg": 4.00, "avg_weight": 0.050, "typical_min_weight": 0.040, "typical_max_weight": 0.060},
    "Avocado": {"price_per_kg": 5.00, "avg_weight": 0.175, "typical_min_weight": 0.150, "typical_max_weight": 0.250},
    "Banana": {"price_per_kg": 1.80, "avg_weight": 0.120, "typical_min_weight": 0.100, "typical_max_weight": 0.150},
    "Beans": {"price_per_kg": 3.00, "avg_weight": 0.150, "typical_min_weight": 0.100, "typical_max_weight": 0.200}, # per pack
    "Beetroot": {"price_per_kg": 2.00, "avg_weight": 0.120, "typical_min_weight": 0.100, "typical_max_weight": 0.150},
    "Blackberry": {"price_per_kg": 12.00, "avg_weight": 0.125, "typical_min_weight": 0.100, "typical_max_weight": 0.150}, # per punnet
    "Blueberry": {"price_per_kg": 15.00, "avg_weight": 0.125, "typical_min_weight": 0.100, "typical_max_weight": 0.150}, # per punnet
    "Cabbage": {"price_per_kg": 1.50, "avg_weight": 1.000, "typical_min_weight": 0.700, "typical_max_weight": 1.500},
    "Cactus fruit": {"price_per_kg": 6.00, "avg_weight": 0.150, "typical_min_weight": 0.100, "typical_max_weight": 0.200},
    "Caju seed": {"price_per_kg": 15.00, "avg_weight": 0.050, "typical_min_weight": 0.040, "typical_max_weight": 0.060}, # per pack of seeds
    "Cantaloupe": {"price_per_kg": 2.20, "avg_weight": 1.200, "typical_min_weight": 1.000, "typical_max_weight": 1.800},
    "Carambula": {"price_per_kg": 7.00, "avg_weight": 0.100, "typical_min_weight": 0.080, "typical_max_weight": 0.120},
    "Carrot": {"price_per_kg": 1.20, "avg_weight": 0.075, "typical_min_weight": 0.060, "typical_max_weight": 0.100},
    "Cauliflower": {"price_per_kg": 2.50, "avg_weight": 0.750, "typical_min_weight": 0.500, "typical_max_weight": 1.000},
    "Cherimoya": {"price_per_kg": 8.00, "avg_weight": 0.300, "typical_min_weight": 0.200, "typical_max_weight": 0.500},
    "Cherry": {"price_per_kg": 9.00, "avg_weight": 0.100, "typical_min_weight": 0.080, "typical_max_weight": 0.150}, # per punnet or small bag
    "Chestnut": {"price_per_kg": 10.00, "avg_weight": 0.020, "typical_min_weight": 0.015, "typical_max_weight": 0.030},
    "Clementine": {"price_per_kg": 3.00, "avg_weight": 0.070, "typical_min_weight": 0.060, "typical_max_weight": 0.090},
    "Cocos": {"price_per_kg": 3.50, "avg_weight": 0.800, "typical_min_weight": 0.600, "typical_max_weight": 1.200}, # Price per item
    "Corn": {"price_per_kg": 2.00, "avg_weight": 0.250, "typical_min_weight": 0.200, "typical_max_weight": 0.300}, # Price per cob
    "Cucumber": {"price_per_kg": 1.80, "avg_weight": 0.350, "typical_min_weight": 0.250, "typical_max_weight": 0.450},
    "Dates": {"price_per_kg": 12.00, "avg_weight": 0.200, "typical_min_weight": 0.150, "typical_max_weight": 0.250}, # per pack
    "Eggplant": {"price_per_kg": 3.00, "avg_weight": 0.400, "typical_min_weight": 0.300, "typical_max_weight": 0.500},
    "Fig": {"price_per_kg": 10.00, "avg_weight": 0.050, "typical_min_weight": 0.040, "typical_max_weight": 0.060},
    "Ginger Root": {"price_per_kg": 7.00, "avg_weight": 0.100, "typical_min_weight": 0.050, "typical_max_weight": 0.150}, # Small piece
    "Gooseberry": {"price_per_kg": 7.50, "avg_weight": 0.125, "typical_min_weight": 0.100, "typical_max_weight": 0.150}, # per punnet
    "Granadilla": {"price_per_kg": 8.50, "avg_weight": 0.060, "typical_min_weight": 0.050, "typical_max_weight": 0.080},
    "Grape": {"price_per_kg": 4.50, "avg_weight": 0.500, "typical_min_weight": 0.400, "typical_max_weight": 0.700}, # per bunch
    "Grapefruit": {"price_per_kg": 2.80, "avg_weight": 0.400, "typical_min_weight": 0.300, "typical_max_weight": 0.500},
    "Guava": {"price_per_kg": 6.50, "avg_weight": 0.100, "typical_min_weight": 0.080, "typical_max_weight": 0.120},
    "Hazelnut": {"price_per_kg": 18.00, "avg_weight": 0.100, "typical_min_weight": 0.080, "typical_max_weight": 0.150}, # per bag shelled
    "Huckleberry": {"price_per_kg": 12.00, "avg_weight": 0.125, "typical_min_weight": 0.100, "typical_max_weight": 0.150}, # per punnet
    "Kaki": {"price_per_kg": 4.00, "avg_weight": 0.180, "typical_min_weight": 0.150, "typical_max_weight": 0.220},
    "Kiwi": {"price_per_kg": 3.50, "avg_weight": 0.090, "typical_min_weight": 0.070, "typical_max_weight": 0.110},
    "Kohlrabi": {"price_per_kg": 2.30, "avg_weight": 0.300, "typical_min_weight": 0.200, "typical_max_weight": 0.400},
    "Kumquats": {"price_per_kg": 9.00, "avg_weight": 0.150, "typical_min_weight": 0.100, "typical_max_weight": 0.200}, # per pack
    "Lemon": {"price_per_kg": 2.20, "avg_weight": 0.100, "typical_min_weight": 0.080, "typical_max_weight": 0.120},
    "Limes": {"price_per_kg": 3.30, "avg_weight": 0.060, "typical_min_weight": 0.050, "typical_max_weight": 0.080},
    "Lychee": {"price_per_kg": 11.00, "avg_weight": 0.200, "typical_min_weight": 0.150, "typical_max_weight": 0.250}, # per pack
    "Mandarine": {"price_per_kg": 3.00, "avg_weight": 0.080, "typical_min_weight": 0.060, "typical_max_weight": 0.100},
    "Mango": {"price_per_kg": 4.50, "avg_weight": 0.300, "typical_min_weight": 0.200, "typical_max_weight": 0.500},
    "Mangostan": {"price_per_kg": 15.00, "avg_weight": 0.100, "typical_min_weight": 0.080, "typical_max_weight": 0.120},
    "Maracuja": {"price_per_kg": 7.50, "avg_weight": 0.080, "typical_min_weight": 0.060, "typical_max_weight": 0.100}, # Same as Passion Fruit
    "Melon Piel de Sapo": {"price_per_kg": 2.00, "avg_weight": 2.000, "typical_min_weight": 1.500, "typical_max_weight": 2.500},
    "Mulberry": {"price_per_kg": 10.00, "avg_weight": 0.125, "typical_min_weight": 0.100, "typical_max_weight": 0.150}, # per punnet
    "Nectarine": {"price_per_kg": 3.80, "avg_weight": 0.140, "typical_min_weight": 0.120, "typical_max_weight": 0.160},
    "Nut Forest": {"price_per_kg": 12.00, "avg_weight": 0.150, "typical_min_weight": 0.100, "typical_max_weight": 0.200}, # Generic mixed nuts pack
    "Nut Pecan": {"price_per_kg": 20.00, "avg_weight": 0.100, "typical_min_weight": 0.080, "typical_max_weight": 0.120}, # per bag shelled
    "Onion": {"price_per_kg": 1.00, "avg_weight": 0.150, "typical_min_weight": 0.100, "typical_max_weight": 0.200},
    "Orange": {"price_per_kg": 2.00, "avg_weight": 0.180, "typical_min_weight": 0.150, "typical_max_weight": 0.220},
    "Papaya": {"price_per_kg": 3.50, "avg_weight": 0.750, "typical_min_weight": 0.500, "typical_max_weight": 1.200},
    "Passion Fruit": {"price_per_kg": 7.50, "avg_weight": 0.080, "typical_min_weight": 0.060, "typical_max_weight": 0.100},
    "Peach": {"price_per_kg": 3.50, "avg_weight": 0.150, "typical_min_weight": 0.130, "typical_max_weight": 0.180},
    "Pear": {"price_per_kg": 2.80, "avg_weight": 0.180, "typical_min_weight": 0.150, "typical_max_weight": 0.220},
    "Pepino": {"price_per_kg": 5.50, "avg_weight": 0.250, "typical_min_weight": 0.200, "typical_max_weight": 0.300},
    "Pepper": {"price_per_kg": 3.20, "avg_weight": 0.150, "typical_min_weight": 0.120, "typical_max_weight": 0.200}, # Bell pepper
    "Physalis": {"price_per_kg": 14.00, "avg_weight": 0.100, "typical_min_weight": 0.080, "typical_max_weight": 0.125}, # per punnet
    "Pineapple": {"price_per_kg": 2.50, "avg_weight": 1.500, "typical_min_weight": 1.000, "typical_max_weight": 2.000},
    "Pistachio": {"price_per_kg": 25.00, "avg_weight": 0.100, "typical_min_weight": 0.080, "typical_max_weight": 0.150}, # per bag shelled
    "Pitahaya": {"price_per_kg": 9.00, "avg_weight": 0.350, "typical_min_weight": 0.250, "typical_max_weight": 0.450},
    "Plum": {"price_per_kg": 4.00, "avg_weight": 0.070, "typical_min_weight": 0.050, "typical_max_weight": 0.090},
    "Pomegranate": {"price_per_kg": 5.00, "avg_weight": 0.300, "typical_min_weight": 0.250, "typical_max_weight": 0.400},
    "Pomelo Sweetie": {"price_per_kg": 3.00, "avg_weight": 1.000, "typical_min_weight": 0.800, "typical_max_weight": 1.200},
    "Potato": {"price_per_kg": 1.20, "avg_weight": 0.200, "typical_min_weight": 0.100, "typical_max_weight": 0.300},
    "Quince": {"price_per_kg": 3.50, "avg_weight": 0.250, "typical_min_weight": 0.200, "typical_max_weight": 0.350},
    "Rambutan": {"price_per_kg": 10.00, "avg_weight": 0.030, "typical_min_weight": 0.020, "typical_max_weight": 0.040},
    "Raspberry": {"price_per_kg": 16.00, "avg_weight": 0.125, "typical_min_weight": 0.100, "typical_max_weight": 0.150}, # per punnet
    "Redcurrant": {"price_per_kg": 10.00, "avg_weight": 0.125, "typical_min_weight": 0.100, "typical_max_weight": 0.150}, # per punnet
    "Salak": {"price_per_kg": 9.50, "avg_weight": 0.050, "typical_min_weight": 0.040, "typical_max_weight": 0.070},
    "Strawberry": {"price_per_kg": 6.00, "avg_weight": 0.250, "typical_min_weight": 0.200, "typical_max_weight": 0.300}, # per punnet
    "Tamarillo": {"price_per_kg": 7.00, "avg_weight": 0.100, "typical_min_weight": 0.080, "typical_max_weight": 0.120},
    "Tangelo": {"price_per_kg": 3.20, "avg_weight": 0.150, "typical_min_weight": 0.120, "typical_max_weight": 0.180},
    "Tomato": {"price_per_kg": 2.80, "avg_weight": 0.120, "typical_min_weight": 0.080, "typical_max_weight": 0.150}, # Standard tomato
    "Walnut": {"price_per_kg": 16.00, "avg_weight": 0.100, "typical_min_weight": 0.080, "typical_max_weight": 0.150}, # per bag shelled
    "Watermelon": {"price_per_kg": 1.00, "avg_weight": 5.000, "typical_min_weight": 3.000, "typical_max_weight": 8.000},
    "Zucchini": {"price_per_kg": 2.00, "avg_weight": 0.200, "typical_min_weight": 0.150, "typical_max_weight": 0.250},
    "Unknown": {"price_per_kg": 0.00, "avg_weight": 0.0, "typical_min_weight": 0.0, "typical_max_weight": 0.0}
} 