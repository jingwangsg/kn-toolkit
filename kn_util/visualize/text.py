import cv2

from .utils import draw_opaque_mask


def draw_text(img,
              point,
              text,
              draw_type="custom",
              font_scale=0.4,
              text_color=(255, 255, 255),
              thickness=5,
              text_thickness=1,
              bg_color=(0, 0, 255)):
    '''
    :param img:
    :param point:
    :param text:
    :param drawType: custom or custom
    :return:
    '''
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    # fontFace=cv2.FONT_HERSHEY_SIMPLEX
    if draw_type == "custom":
        text_size, baseline = cv2.getTextSize(str(text), font_face, font_scale, thickness)
        text_loc = (point[0], point[1] + text_size[1])
        if bg_color == "opaque":
            img = draw_opaque_mask(img, (text_loc[0] - 2 // 2, text_loc[1] - 2 - baseline),
                                   (text_loc[0] + text_size[0], text_loc[1] + text_size[1]),
                                   alpha=0.5)
        else:
            cv2.rectangle(img, (text_loc[0] - 2 // 2, text_loc[1] - 2 - baseline),
                          (text_loc[0] + text_size[0], text_loc[1] + text_size[1]), bg_color, -1)
        # draw score value
        cv2.putText(img, str(text), (text_loc[0], text_loc[1] + baseline), font_face, font_scale, text_color,
                    text_thickness, 8)

    elif draw_type == "simple":
        cv2.putText(img, '%d' % (text), point, font_face, 0.5, (255, 0, 0))
    return img


def draw_text_line(img,
                   point,
                   text_line: str,
                   draw_type="custom",
                   text_color=(255, 255, 255),
                   font_scale=0.4,
                   thickness=5,
                   text_thickness=1,
                   bg_color=(0, 0, 255)):
    '''
    :param img:
    :param point:
    :param text:
    :param drawType: custom or custom
    :return:
    '''
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    # fontFace=cv2.FONT_HERSHEY_SIMPLEX
    text_line = text_line.split("\n")
    # text_size, baseline = cv2.getTextSize(str(text_line), fontFace, fontScale, thickness)
    text_size, baseline = cv2.getTextSize(str(text_line), font_face, font_scale, thickness)
    for i, text in enumerate(text_line):
        if text:
            draw_point = [point[0], point[1] + (text_size[1] + 2 + baseline) * i]
            img = draw_text(img,
                            draw_point,
                            text,
                            draw_type,
                            thickness=thickness,
                            text_thickness=text_thickness,
                            bg_color=bg_color,
                            text_color=text_color)
    return img
