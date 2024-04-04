import numpy as np
import cv2
from scipy import stats
from math import atan2, degrees

kpt_names_index = {
        'nose':0, 'neck':1,
        'r_sho':2, 'r_elb':3, 'r_wri':4, 'l_sho':5, 'l_elb':6, 'l_wri':7,
        'r_hip':8, 'r_knee':9, 'r_ank':10, 'l_hip':11, 'l_knee':12, 'l_ank':13,
        'r_eye':14, 'l_eye':15,'r_ear':16, 'l_ear':17}


def angle_between_points(p1, p2, p3):
    """
    Calculate the angle between two lines formed by three points p1, p2, and p3.
    The lines are formed between p1-p2 and p2-p3.
    
    Args:
    - p1, p2, p3: Lists representing the x, y coordinates of the points.
    
    Returns:
    - The angle in degrees between the two lines.
    """

    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])

    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    
    angle = np.arccos(dot_product) * 180 / np.pi

    return 180.0-angle


def set_standard(standards):
    """
    Get points, standard angle and allowed error from standards dict.
    Args:
        standards: Dictionary, {"p1-p2-p3":[standard angle, allowed error]}, 
                   e.g. {"2-3-4":[180, 10]}
    Returns:

    """
    angle_points = [key for key in standards]
    standard_error = [val for val in standards.values()]
    return angle_points, standard_error


def get_angle(body_points, angle_points):
    """
    Get the angle of keypoints that take into consideration.
    args:
        body_points: a 2-dimension list of the predicted position of all body points.
        angle_points: a list of input angles(string), in the format of 'p1-p2-p3'.
    returns:
        pose_angles: the predicted angles of the keypoints that take into consideration.
    """
    pose_angles = []
    for i in range(len(angle_points)):
        points = list(map(int, angle_points[i].split('-')))
        p1, p2, p3 = body_points[points[0]], body_points[points[1]], body_points[points[2]]
        # len(p2) == 2  

        #Calculate angle between p1 and p2 and p3 (p1-p2 & p2-p3)
        angle = angle_between_points(p1, p2, p3)
        pose_angles.append(angle)
    return pose_angles


def cal_score(body_points, standards, alpha=9):
    """
    Calculate the score of the given standards and predicted angles

    args:
        body_points: the predicted keypoints by openpose
        stanards: standard angle and allowed error from input
        alpha: the allowed error space, max_error = error*alpha
    returns:
        scores: a list of the score of each angle.
    """

    # input process
    angle_points, standard_error = set_standard(standards)
    pose_angles = get_angle(body_points, angle_points)

    # get the standard angles and errors
    standard_angles = []
    errors = []
    for p in range(len(standard_error)):
        standard_angles.append(standard_error[p][0])
        errors.append(standard_error[p][1])
    print(f"standard_angles: {standard_angles}")

    # scores
    assert len(pose_angles) == len(standard_angles)

    scores = [] # calculate the score of each angle
    for i in range(len(standard_angles)):
        if pose_angles[i] <= standard_angles[i] + errors[i] and pose_angles[i]>=standard_angles[i]-errors[i]:
            scores.append(100)
        else:
            abs_diff = abs(pose_angles[i]-standard_angles[i])
            print(f"abs_diff:{abs_diff}")
            max_diff = errors[i]*alpha
            diff = abs_diff - errors[i]

            # if the pose not in max_diff, put "danger" or "not standard!" warning! 
            if abs_diff >= max_diff:
                draw_danger(i, pose_angles[i])
            # score decreased from 100-1
            score_decrement = (abs_diff / (max_diff - errors[i])) * 99 
            scores.append(max(100 - score_decrement, 1)) 
    scores = [round(num, 2) for num in scores] 
    return scores


def draw_scores(scores, img, standards, bbox):
    """
    Show the score to the img.
    args:
        scores: the list of each angle's score
        img: the one that will be shown. 
        standards: get the keypoints that take into consideration.
        bbox: bounding box for each people
    """
    angle_points, _ = set_standard(standards)
    assert len(angle_points) == len(scores)

    # shown setting
    text_sample = "AVG Score: 78.91"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.75
    color = (255, 255, 0)
    thickness = 2

    # text space
    (text_width, text_height), _ = cv2.getTextSize(text_sample, font, fontScale, thickness)

    margin = 20  # margin distance to the image edge
    # start_x = img.shape[1] - text_width - margin 
    # start_y = text_height + margin
    start_x = bbox[0] + bbox[2] - text_width
    start_y = bbox[1] - (text_height+margin)*(len(scores)+1)

    # Overall score
    cv2.putText(img, f"AVG Score: {sum(scores)/len(scores): .2f}", (start_x, start_y), font, fontScale, color, thickness)
    
    # Each angle score
    for i in range(len(scores)):
        name = angle_points[i]
        start_y += text_height + margin
        cv2.putText(img, f"{name}: {str(scores[i])}", (start_x, start_y), font, fontScale, color, thickness)
    

def draw_danger(points_index, angle):
    pass

def input_img_standards(img):
    pass

if __name__ == "__main__":
    pass