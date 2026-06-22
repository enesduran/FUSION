import re
import json 
import itertools
 
self_interaction_actions = {

    "finger_to_hand": [   
        "Cracking knuckles",
        "Interlacing fingers",
        "Fist bump your hands",
        "Make delicious gesture",
        "Rubbing the palms together",
        "Rubbing fingertips together",
        "Pressing thumb against palm",
        "Rolling a ring around a finger",
        "Flicking one finger with another",
        "Massaging palm with opposite hand",
        "Tapping fingers against each other",
        "Snap your fingers to make some noise",
        "Holding one wrist with the opposite hand",
        "Pulling fingers one by one with the opposite hand",
        "Drumming left fingers on the back of the right hand",
        "Tracing a circle or shape on the palm with your thumb",
        ],
    
    "finger_to_face": [
        "Smell your armpit",
        "Scratch your eyes",
        "Scratching forehead", 
        "Rub your back of head",
        "Blow nose with your right",
        "Tapping cheek with fingers",
        "Scratching chin or jawline",
        "Wiping sweat from forehead",
        "Scratch your eyes with thumbs",
        "Gently pinching bridge of nose",
        "Rubbing temples with fingertips",
        "Resting fingertips against forehead",
        "Running ring and middle fingertips across lips",
        "Gently pinching and releasing fingers together",
        "Cover your mouth with your left hand over surprising news",
        ],
       
    "hand_to_body": [
        "Fixing hair",
        "Crossing arms",
        "Touch you ear",
        "Grab your ear",
        "Cover both ears",
        "Scratching head",
        "Holding stomach",
        "Adjusting a tie",
        "Hugging yourself",
        "Pinching own skin",
        "Adjusting glasses",
        "Gripping shoulder",
        "Rubbing your knee", 
        "Tighten your belt",
        "Rub your left calf",
        "Salute a commander",
        "Kiss your hand palm",
        "Button your trousers",
        "Adjusting a necklace",
        "Rub your outer elbow",
        "Clapping hands twice",
        "Resting hand on chin",
        "Place hand over heart",
        "Rubbing hands together",
        "Grab your opposite ear",
        "Gently touch your back",
        "Gently scratch your shin", 
        "Stay in attention position",
        "Put your hands on your hips",
        "Adjusting a tie or necklace",
        "Both hands grabbing earlobes",
        "Send a kiss to your valentine",
        "Running fingers along the arm",
        "Zipping or buttoning a jacket",
        "Brushing off dust from clothes",
        "Place your hands on your waists",
        "Roll up both sleeves one by one",
        "Trace a circle around your belly",
        "Touching chin with the left hand",
        "Crossing the arms over the chest",
        "Rubbing belly with your left palm", 
        "Fixing sleeves or rolling them up",
        "Clap your belly with your left hand",
        "Hands on knees to rest after running",
        "Clean your armpit with your left hand",
        "Scretch yourself by touching your toes",
        "Touch your left heel with the left hand",
        "Resting one hand on the opposite shoulder",
        "Crossing arms and tapping arm with fingers",
        "Placing both hands in front of the abdomen",
        "Wrapping fingers around the opposite wrist",
        "Clean your right armpit with your left hand",
        "Touching the back of the neck with one hand",
        "Secure your head in the moment of emergency",
        "Pressing fingers against biceps or shoulders",
        "Scratching forearm or wrist with opposite fingers",
        "Tracing a shape on own forearm with the right index finger",
        "Making a 'spider walk' motion with index and middle fingertips on the opposite forearm",
        ],
    
    "leg_to_leg": [
        "Do a squat",
        "Cross your legs",
        "One heel to the other knee",
    ]
    }
    

examples_new = {
    1: {
        'text': 'Touch your index with your thumb of your left hand.',
        'command_list': [
            [[20, 120], [[4933, 5361]]] 
        ]
    },
    2:  {
        'text': "Gently scratching neck.",
        'command_list': [
            [[40, 50], [[4903, 9006], [5001, 5617], [4974, 5484]]],
            [[52, 63], [[5001, 9006], [4974, 5617], [5068, 5484]]],
            [[66, 77], [[4974, 9006], [5068, 5617], [5070, 5484]]],
            [[80, 90], [[5068, 9006], [5070, 5617], [5082, 5484]]]
        ] 
    },
    3: {
        'text': 'Do the ok gesture with the right hand.',
        'command_list': [
            [[30, 60], [[7795, 8095]]],  
        ]
    },
    4: {
        'text': 'Touch your forehead with the left hand like you are thinking.',
        'command_list': [
            [[30, 55], [[5298, 9172], [5287, 9172], [5082, 9234], [5070,9234 ], [5361, 705]]],  
            [[60, 120], [[5298, 705], [5287, 705], [5082, 2161], [5070,2161 ], [5361, 705]]]
        ]
    }
  }


def convert_command_list_to_dict(input_text, identifier=11):
    """
    Converts a generated command list string into a structured Python dictionary.
    
    Args:
        input_text (str): The input string in the format:
            ===== Generated Command List for 'action' ===== Step 1: Time frame [x, y], Vertices: [[a, b, c]] ...
        identifier (int, optional): An identifier number for the output. Defaults to 11.
    
    Returns:
        dict: A dictionary representation of the command list.
    """
    # Extract the action name
    title_start = input_text.find("'") + 1
    title_end = input_text.find("'", title_start)
    action_text = "rubbing eyes"  # Default in case not found
    
    if title_start > 0 and title_end > title_start:
        action_text = input_text[title_start:title_end]
    
    # Parse steps
    command_list = []
    
    # Split the input by "Step" to get each step
    steps = input_text.split("Step")[1:]
    
    for step in steps:
        # Extract time frame
        time_frame_start = step.find("[") + 1
        time_frame_end = step.find("]")
        time_frame = step[time_frame_start:time_frame_end].split(", ")
        time_frame = [int(t) for t in time_frame]
        
        # Extract vertices
        vertices_str = step[step.find("Vertices: ") + 10:].strip()
        # Find the array inside the string by matching everything between [[ and ]]
        vertices_match = vertices_str.split("[[")[1].split("]]")[0] if "[[" in vertices_str else ""
        
        vertices = []
        if vertices_match:
            # Split by commas and convert to integers
            vertices = [int(num.strip()) for num in vertices_match.split(',')]
        
        # Add to command list
        command_list.append([time_frame, [vertices]])
    
    # Create the dictionary
    result_dict = {
        identifier: {
            "text": action_text,
            "command_list": command_list
        }
    }
    
    return result_dict

def create_self_interaction_evaluation_prompt(new_gesture_text):
    
    print('gesture', new_gesture_text)
 
    vertex_ids = json.dumps(smplx_vertex_ids, indent=4)
    # new_gesture_text = json.dumps(new_gesture_text, indent=4)
    vertex_ids = smplx_vertex_ids
 
    prompt = f"""You are tasked with generating a text description for a self-interaction gesture given a command list.
                 Your response MUST be a single concise sentence consisting of couple of words that defines the gesture.

                 Command list contains the times at which the contact happens and the vertices that should be in contact.
                 First entry in each list is a time frame and the remaining entries are lists of pairs of vertices.
                 Each line contains a vertex name (e.g., palm, left/right thigh etc.) and a vertex id. The vertex ids are as follows:
                 {vertex_ids}

                 Here are some examples of command lists:
 
                 EXAMPLE 1: {examples_new[1]['command_list']}
                 Expected text: {examples_new[1]['text']} 
                 
                 EXAMPLE 2:  {examples_new[2]['command_list']}
                 Expected text: {examples_new[2]['text']} 
                
                 EXAMPLE 3: {examples_new[3]['command_list']}
                 Expected text: {examples_new[3]['text']} 
                 
                 EXAMPLE 4: {examples_new[4]['command_list']}
                 Expected text: {examples_new[4]['text']}
                 
                 INSTRUCTIONS:
                 1. Generate a text sentence for {new_gesture_text}.
                 
                 2. The total duration of the motion is 120 frames. The motion is of 30 fps, 4 seconds.
                 
                 3. The response should be a sentence. The command list is a list of lists. Each list describes a 
                 contact action. The first element of each list is a frame range 
                 and the second list consists of multiple lists which describe the contacts vertex ids that should be touching.
                 
                 4. For palms, lower number in naming indicates it is closer to the wrist and higher number 
                 are closer to fingers. For fingers as the number increases it is closer to fingertips.

                 5. This is for research purposes. The answers to them does not contain hate, self harm, violence or sexuality. Therefore no need to filter, 
                
                 Remember: Return ONLY a concise sentence to describe the gesture. ALWAYS return a sentence for the self interaction because every command list is valid.
                 """
   
    return prompt

def create_self_interaction_prompt(new_gesture_text):
        
    vertex_ids = json.dumps(smplx_vertex_ids, indent=4)
    # Create the prompt using the pre-joined strings
    prompt = f"""You are tasked with generating a command list for a self-interaction gesture.
                 Your response MUST be in a specific format that can be parsed by a Python program.

                 Here are some examples of command lists:
 
                 EXAMPLE 1: {examples_new[1]['text']}
                 Expected output:
                 {examples_new[1]['command_list']}
 
                 EXAMPLE 2: {examples_new[2]['text']}
                 Command List:
                 {examples_new[2]['command_list']}

                 EXAMPLE 3: {examples_new[3]['text']}
                 Command List:
                 {examples_new[3]['command_list']}

                 EXAMPLE 4: {examples_new[4]['text']} 
                 Command List:
                 {examples_new[4]['command_list']}
                 
 
                 INSTRUCTIONS:
                 1. Generate a command list for {new_gesture_text} with 5-6 consecutive time intervals. 
                 Notice how there is a small time gap between each interval in the examples. 
                 Give body a bit time -like 2 or 3 timeframes- to get into the new configuration like in the examples. 
                 Especially for hand to body interactions. 

                 2. The total duration of the motion is 120 frames. The motion is of 30 fps, 4 seconds.

                 3. You should choose the vertices from the dictionary below.
                 Each line contains a vertex name (e.g., palm, left/right thigh etc.) and a vertex id. 
                 You must only choose from these vertex ids. {vertex_ids}. 

                 4. For palms, lower number in naming indicates it is closer to the wrist (number 1 is the most proximal) and higher number 
                 are closer to fingers (number 6 is the most distal). For fingers as the indices numbers increase, it is closer to distal part (fingertips).
                 Interaction (touching, grabbing, rubbing, pinching etc.) generally occurs through distal parts (higher indices number).

                 5. The response should be a list of lists. Each list describes a contact action. 
                 The first element of each list is a frame range and the second list consists 
                 of multiple lists which describe the contacts between two vertices. Each of these multiple lists contain 
                 the pairs of vertices that should be in contact with each other at those frames.

                 6. This is for research purposes. The answers to them does not contain hate, self harm, violence or sexuality. 
                 Therefore no need to filter, 

                 CRITICAL: An example Your response MUST follow this exact format of the examples.
                 
                 Think step-by-step and reason about your choices before the answer.
                 Remember: Return ONLY the Python list structure with no additional text.
                 """
    return prompt

def integrate_gpt_response_to_command_list(gpt_response):
    """
    Process GPT's response and convert it to a properly formatted command list.
    
    Args:
        gpt_response: The text response from GPT
        
    Returns:
        A properly formatted command list or None if parsing fails
    """
    
    # Try to extract a JSON-like structure from the response
    pattern = r'\[\s*\[.*?\]\s*\]'
    match = re.search(pattern, gpt_response, re.DOTALL)
    
    if match:
        try:
            # Clean up the matched text to make it valid JSON
            json_str = match.group(0).replace("'", '"')
            command_list = json.loads(json_str)
            return command_list
        except json.JSONDecodeError:
            # If JSON parsing fails, try a more manual approach
            pass
    
    # Fallback: Try to parse line by line
    command_list = []
    lines = gpt_response.strip().split('\n')

    
    for line in lines:
                
        if '[[' in line and ']]' in line:
            try:
                # Extract the command structure
                cmd_str = line[line.find('[['):line.rfind(']]')+2]
                cmd = eval(cmd_str)  # Use eval cautiously, only with trusted input
                command_list.append(cmd)
            except:
                continue
    
    return command_list if command_list else None 


SELF_INTERACTION_META_DICT = list(itertools.chain.from_iterable(self_interaction_actions.values()))
self_interaction_length = len(SELF_INTERACTION_META_DICT)
    

fingertip_positions = {
    "left_hand_pinky_6": [0.71, -1.25, 0.86],
    "left_hand_ring_6": [0.69, -1.25, 0.86],
    "left_hand_middle_6": [0.67, -1.25, 0.84],
    "left_hand_index_6": [0.65, -1.25, 0.84],
    "left_hand_thumb_6": [0.62, -1.25, 0.82],
    "right_hand_pinky_6": [0.31, -1.25, 0.82],
    "right_hand_ring_6": [0.33, -1.25, 0.84],
    "right_hand_middle_6": [0.35, -1.25, 0.84],
    "right_hand_index_6": [0.37, -1.25, 0.86],
    "right_hand_thumb_6": [0.39, -1.25, 0.86]
    }


smplx_vertex_hands_ids = {
    'left_hand_pinky_1': 5221,
    'left_hand_pinky_2': 5229,
    'left_hand_pinky_3': 5198,
    'left_hand_pinky_4': 5296,
    'left_hand_pinky_5': 5298,
    'left_hand_pinky_6': 5287,
    
    'left_hand_ring_1': 4745,
    'left_hand_ring_2': 5111,
    'left_hand_ring_3': 5086,
    'left_hand_ring_4': 5179,
    'left_hand_ring_5': 5181,
    'left_hand_ring_6': 5170,
    
    'left_hand_middle_1': 4903,
    'left_hand_middle_2': 5001,
    'left_hand_middle_3': 4974,
    'left_hand_middle_4': 5068,
    'left_hand_middle_5': 5070,
    'left_hand_middle_6': 5082,
    
    'left_hand_index_1': 4776,
    'left_hand_index_2': 4800,
    'left_hand_index_3': 4644,
    'left_hand_index_4': 4956,
    'left_hand_index_5': 4958,
    'left_hand_index_6': 4933,
    
    'left_hand_thumb_1': 4881,
    'left_hand_thumb_2': 5371,
    'left_hand_thumb_3': 5373,
    'left_hand_thumb_4': 5384,
    'left_hand_thumb_5': 5362,
    'left_hand_thumb_6': 5361,

    'right_hand_pinky_1': 7957,
    'right_hand_pinky_2': 7965,
    'right_hand_pinky_3': 7934,
    'right_hand_pinky_4': 8032,
    'right_hand_pinky_5': 8034,
    'right_hand_pinky_6': 8046,
    
    'right_hand_ring_1': 7839,
    'right_hand_ring_2': 7819,
    'right_hand_ring_3': 7858,
    'right_hand_ring_4': 7916,
    'right_hand_ring_5': 7924,
    'right_hand_ring_6': 7906,
    
    'right_hand_middle_1': 7730,
    'right_hand_middle_2': 7707,
    'right_hand_middle_3': 7748,
    'right_hand_middle_4': 7805,
    'right_hand_middle_5': 7813,
    'right_hand_middle_6': 7795,
    
    'right_hand_index_1': 7511,
    'right_hand_index_2': 7377,
    'right_hand_index_3': 7583,
    'right_hand_index_4': 7693,
    'right_hand_index_5': 7701,
    'right_hand_index_6': 7669,
    
    'right_hand_thumb_1': 7617,
    'right_hand_thumb_2': 8105,
    'right_hand_thumb_3': 8107,
    'right_hand_thumb_4': 8114,
    'right_hand_thumb_5': 8119,
    'right_hand_thumb_6': 8095,

    'right_pinky_palm_1': 7471,
    'right_pinky_palm_2': 7430,
    'right_pinky_palm_3': 8123,
    'right_pinky_palm_4': 8121,
    
    'right_ring_palm_1': 7605,
    'right_ring_palm_2': 7530,
    'right_ring_palm_3': 7401,
    'right_ring_palm_4': 7482,
    
    'right_middle_palm_1': 7442,
    'right_middle_palm_2': 7426,
    'right_middle_palm_3': 7399,
    'right_middle_palm_4': 7621,
    
    'right_index_palm_1': 7444,
    'right_index_palm_2': 7333,
    'right_index_palm_3': 7396,
    'right_index_palm_4': 7512,

    'left_pinky_palm_1': 4735,
    'left_pinky_palm_2': 4694,
    'left_pinky_palm_3': 5389,
    'left_pinky_palm_4': 5387,
    
    'left_ring_palm_1': 4628,
    'left_ring_palm_2': 4851,
    'left_ring_palm_3': 4667,
    'left_ring_palm_4': 4750,
    
    'left_middle_palm_1': 4707,
    'left_middle_palm_2': 4899,
    'left_middle_palm_3': 4664,
    'left_middle_palm_4': 4882,
    
    'left_index_palm_1': 4709,
    'left_index_palm_2': 4598,
    'left_index_palm_3': 4655,
    'left_index_palm_4': 4660,
}


smplx_vertex_ids = {
    'left_upper_scapula': 5463, 
    'left_lower_scapula': 5565,
    'right_upper_scapula': 8197,
    'right_lower_scapula': 6128,
    
    'neck_upper': 9006, 
    'neck_middle': 5617, 
    'neck_below': 5484, 
    
    'left_upper_forehead': 2161,
    'left_lower_forehead': 705,
    'middle_upper_forehead': 8964,
    'middle_lower_forehead': 8968,
    'right_upper_forehead': 9234,
    'right_lower_forehead': 9172,

    'upper_back_of_the_head': 8967,
    'middle_back_of_the_head': 8966,
    'lower_back_of_the_head': 8980,
    
    'left_upper_buttock': 5686,
    'left_middle_buttock': 5694,
    'left_lower_buttock': 5666,
    'right_upper_buttock': 8383,
    'right_middle_buttock': 8376,
    'right_lower_buttock': 8360, 
    
    'left_upper_back_thigh': 3484,
    'left_middle_back_thigh': 4092,
    'left_lower_back_thigh': 3802,
    'right_upper_back_thigh': 6245,
    'right_middle_back_thigh': 6836,
    'right_lower_back_thigh': 6560,
    
    'nasal_bump': 8960,
    'glabella': 9156,
    'middle_lower_nose': 8945, 
    'right_lower_nose': 3102,
    'left_lower_nose': 2066,
    
    'right_eyebrow_rightmost': 337,
    'right_eyebrow_middle': 3154,
    'right_eyebrow_leftmost': 9136,
    'left_eyebrow_rightmost': 9303,
    'left_eyebrow_middle': 27,
    'right_eyebrow_leftmost': 1429,

    'right_eye_cornea': 10049,
    'left_eye_cornea': 9503,
    
    'top_head': 9011,
    'left_temple_head': 802,
    'right_temple_head': 3064,
    
    'right_upper_lip': 2850,
    'middle_upper_lip': 8990,
    'left_upper_lip': 1735,
    'right_lower_lip': 2882,
    'middle_lower_lip': 8950,
    'left_lower_lip': 1794,
    
    'right_chin': 8757,
    'middle_chin': 8755,
    'left_chin': 9066,
    
    'left_inner_earlobe': 47,
    'left_outer_earlobe': 40,
    'right_inner_earlobe': 587,
    'right_outer_earlobe': 981,
    
    'right_cheek': 3124,
    'left_cheek': 2102,
    
    'left_heel': 8846, 
    'right_heel': 8634,
    'left_lower_calf': 3763,
    'left_upper_calf': 4098,
    'right_upper_calf': 6842, 
    'right_lower_calf': 6521,
    
    'left_big_toe': 5770,
    'left_small_toe': 5780,
    'right_big_toe': 8463,
    'right_small_toe': 8474,
    
    'left_hip': 3865,
    'right_hip': 6616, 
    
    'right_kneecap': 6438,
    'right_upper_shin': 6494,
    'right_middle_shin': 6468,
    'right_lower_shin': 8569,
    'left_kneecap': 3677, 
    'left_upper_shin': 3733,
    'left_middle_shin': 3707,
    'left_lower_shin': 5875,
    
    'right_upper_thigh_front': 6267,
    'right_middle_thigh_front': 6338,
    'right_lower_thigh_front': 6577,
    'left_upper_thigh_front': 3506,
    'left_middle_thigh_front': 3577,
    'left_lower_thigh_front': 3820,
    
    'lower_belly_button': 4291,
    'middle_belly_button': 5939,
    'upper_belly_button': 5943,
    'left_belly_button': 3546,
    'right_belly_button': 6307,

    'lower_pubic_area': 5601,
    'middle_pubic_area': 5949,
    'upper_pubic_area': 4320,

    'thoracic_spine_T6': 5947,
    'thoracic_spine_T7': 5487,
    'thoracic_spine_T8': 5486,
    'thoracic_spine_T9': 5944,

    'left_waist_front': 5514,
    'left_waist_back': 5506,
    'right_waist_front': 8236,
    'right_waist_back': 8228,
    'lumbar_spine_L3': 4298,

    'upper_breastbone': 5528,
    'middle_breastbone': 5936,
    'lower_breastbone': 5531,
    
    'right_nipple': 8329,
    'left_nipple': 3985,
 
    'right_armpit_back': 6076,
    'right_armpit_middle': 6919,
    'right_armpit_front': 6033,
    
    'left_armpit_back': 3923,
    'left_armpit_middle': 4169,
    'left_armpit_front': 3302,
    
    'right_biceps': 6022,
    'right_elbow_inside': 7026,
    'right_elbow_backside': 7032,
    'right_forearm_1': 6948,
    'right_forearm_2': 6953,
    'right_forearm_3': 6962,
    
    'left_biceps': 5397,
    'left_elbow_inside': 4284,
    'left_elbow_backside': 4293,
    'left_forearm_1': 4204,
    'left_forearm_2': 4218,
    'left_forearm_3': 4571,
    
    'right_shoulder_front': 7217,
    'right_shoulder_top': 6629,
    'right_shoulder_back': 8199,
    'left_shoulder_front': 4480,
    'left_shoulder_top': 5627,
    'left_shoulder_back': 4469,

    'left_wrist_dorsal': 4823,
    'left_wrist_palmar': 4717,
    'right_wrist_dorsal': 7558,
    'right_wrist_palmar': 7449,
    
    'left_hand_pinky_1': 5221,
    'left_hand_pinky_2': 5229,
    'left_hand_pinky_3': 5198,
    'left_hand_pinky_4': 5296,
    'left_hand_pinky_5': 5298,
    'left_hand_pinky_6': 5287,
    
    'left_hand_ring_1': 4745,
    'left_hand_ring_2': 5111,
    'left_hand_ring_3': 5086,
    'left_hand_ring_4': 5179,
    'left_hand_ring_5': 5181,
    'left_hand_ring_6': 5170,
    
    'left_hand_middle_1': 4903,
    'left_hand_middle_2': 5001,
    'left_hand_middle_3': 4974,
    'left_hand_middle_4': 5068,
    'left_hand_middle_5': 5070,
    'left_hand_middle_6': 5058,
    
    'left_hand_index_1': 4776,
    'left_hand_index_2': 4800,
    'left_hand_index_3': 4644,
    'left_hand_index_4': 4956,
    'left_hand_index_5': 4958,
    'left_hand_index_6': 4933,
    
    'left_hand_thumb_1': 4881,
    'left_hand_thumb_2': 5371,
    'left_hand_thumb_3': 5373,
    'left_hand_thumb_4': 5384,
    'left_hand_thumb_5': 5362,
    'left_hand_thumb_6': 5361,

    'left_hand_pinky_knuckle': 4807,
    'left_hand_ring_knuckle': 4907,
    'left_hand_middle_knuckle': 4828,
    'left_hand_index_knuckle': 4747,
    'left_hand_thumb_knuckle': 4841,

    'right_hand_pinky_1': 7957,
    'right_hand_pinky_2': 7965,
    'right_hand_pinky_3': 7934,
    'right_hand_pinky_4': 8032,
    'right_hand_pinky_5': 8034,
    'right_hand_pinky_6': 8046,
    
    'right_hand_ring_1': 7839,
    'right_hand_ring_2': 7819,
    'right_hand_ring_3': 7858,
    'right_hand_ring_4': 7916,
    'right_hand_ring_5': 7924,
    'right_hand_ring_6': 7906,
    
    'right_hand_middle_1': 7730,
    'right_hand_middle_2': 7707,
    'right_hand_middle_3': 7748,
    'right_hand_middle_4': 7805,
    'right_hand_middle_5': 7813,
    'right_hand_middle_6': 7795,
    
    'right_hand_index_1': 7511,
    'right_hand_index_2': 7377,
    'right_hand_index_3': 7583,
    'right_hand_index_4': 7693,
    'right_hand_index_5': 7679,
    'right_hand_index_6': 7669,
    
    'right_hand_thumb_1': 7617,
    'right_hand_thumb_2': 8105,
    'right_hand_thumb_3': 8107,
    'right_hand_thumb_4': 8114,
    'right_hand_thumb_5': 8119,
    'right_hand_thumb_6': 8095,

    'right_hand_pinky_knuckle': 7412,
    'right_hand_ring_knuckle': 7525,
    'right_hand_middle_knuckle': 7564,
    'right_hand_index_knuckle': 7483,
    'right_hand_thumb_knuckle': 7577,

    'right_pinky_palm_1': 7471,
    'right_pinky_palm_2': 7430,
    'right_pinky_palm_3': 8123,
    'right_pinky_palm_4': 8121,
    
    'right_ring_palm_1': 7605,
    'right_ring_palm_2': 7530,
    'right_ring_palm_3': 7401,
    'right_ring_palm_4': 7482,
    
    'right_middle_palm_1': 7442,
    'right_middle_palm_2': 7426,
    'right_middle_palm_3': 7399,
    'right_middle_palm_4': 7621,
    
    'right_index_palm_1': 7444,
    'right_index_palm_2': 7333,
    'right_index_palm_3': 7396,
    'right_index_palm_4': 7395,

    'left_pinky_palm_1': 4735,
    'left_pinky_palm_2': 4694,
    'left_pinky_palm_3': 5389,
    'left_pinky_palm_4': 5387,
    
    'left_ring_palm_1': 4628,
    'left_ring_palm_2': 4851,
    'left_ring_palm_3': 4667,
    'left_ring_palm_4': 4750,
    
    'left_middle_palm_1': 4707,
    'left_middle_palm_2': 4899,
    'left_middle_palm_3': 4664,
    'left_middle_palm_4': 4882,
    
    'left_index_palm_1': 4709,
    'left_index_palm_2': 4598,
    'left_index_palm_3': 4655,
    'left_index_palm_4': 4660,

}