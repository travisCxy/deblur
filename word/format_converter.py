# coding:utf-8
from word import rotated_rect_utils
from word.type_cls import TypeCls
import copy


def find_topic_item(topic_items, sub_item, thresh=0.7):
    if len(topic_items) == 0:
        return []
    sub_item_box = sub_item["region"] + [sub_item["rotation"]]
    ratios = [rotated_rect_utils.rotated_rect_contains_ratio(
        topic["region"] + [topic["rotation"]], sub_item_box) for topic in topic_items]

    ret = []
    for topic, ratio in zip(topic_items, ratios):
        if ratio > thresh:
            ret.append(topic)
    return ret


def append_item_to_topic(topic, sub_item, key):
    items = topic.get(key, [])
    items.append(sub_item)
    topic[key] = items


"""

def fix_one_line_topic_issue(outter_regions, outter_clses):

    #单行的题目可能会导致模型输出时仅输出1个topic或者line区域
    #我们需要检查出这种情况， 把它转化为2个topic和line的区域
    #1. 单个line或者topic悬空
    #2. 这个region不和其他的line/topic交叉
    #3. 有answers落在这个region内部

    new_areas = []
    all_items = zip(outter_regions, outter_clses)

    topic_items = []
    line_items = []
    answer_items = []
    picture_items = []

    for item in all_items:
        if item[1] == TypeCls.outter_topic:
            topic_items.append(item)
        elif item[1] == TypeCls.outter_line:
            line_items.append(item)
        elif item[1] == TypeCls.outter_handwrite:
            answer_items.append(item)
        elif item[1] == TypeCls.outter_picture:
            picture_items.append(item)
    
    def find_parent(candidates, sub_item):
        for canddiate in candidates:
            if rotated_rect_utils.rotated_rect_contains(canddiate[0], sub_item[0]): 
                return canddiate
        return None

    line_without_topic = []
    topic_with_line = []
    topic_with_picture = []
    topic_or_line_with_answer = []
    for line in line_items:
        topic = find_parent(topic_items, line)
        if topic == None:
            line_without_topic.append(line)
        else:
            topic_with_line.append(topic)
    for picture in picture_items:
        topic = find_parent(topic_items, picture)
        if topic != None:
            topic_with_picture.append(topic)
    for answer in answer_items:
        item = find_parent(topic_items + line_items, answer)
        if item != None:
            topic_or_line_with_answer.append(item)

    dangle_lines = line_without_topic
    dangle_topics = [x for x in topic_items if x not in (topic_with_line + topic_with_picture)]
    for dangle_item in dangle_lines + dangle_topics:
        dangle_region = dangle_item[0]
        has_cross_region = False
        for item in all_items:
            region = item[0]
            if item != dangle_item and item[1] in [TypeCls.outter_line, TypeCls.outter_topic]:
                if rotated_rect_utils.rotated_rect_contains(dangle_region, region, thresh = 0.2):
                    has_cross_region = True
                    break

        if not has_cross_region:
            if dangle_item in topic_or_line_with_answer:
                print("found mixed topic/line one line question %d,%d,%d,%d" % tuple(dangle_region[:4]))
                new_area = [[x for x in dangle_item[0]], 0]
                new_area[1] = TypeCls.outter_topic if dangle_item[1] == TypeCls.outter_line else TypeCls.outter_line
        
                new_areas.append(new_area)

    for new_area in new_areas:
        outter_regions.append(new_area[0])
        outter_clses.append(new_area[1])

    return outter_regions, outter_clses
"""


def to_topic_json_format(raw_data, throw_when_no_topic_found=False):

    topic_items = []
    sub_items = []
    for item in raw_data:
        cls = item["cls"]
        if cls in TypeCls.topic_clses:
            topic = item.copy()
            if cls == TypeCls.TOPIC_CALC:
                topic['type'] = 'calc'
            topic_items.append(topic)
        elif cls != TypeCls.STUDENT:
            sub_items.append(item)
    dangle_items = []
    for sub_item in sub_items:
        thresh = 0.1 if sub_item["cls"]==TypeCls.ANSWER else 0.6
        topics = find_topic_item(topic_items, sub_item, thresh)
        if "result" in sub_item:
            sub_item = copy.deepcopy(sub_item)
        for topic in topics:
            sub_item_cls = sub_item["cls"]
            if sub_item_cls == TypeCls.PICTURE:
                append_item_to_topic(topic, sub_item, "pictures")
            elif sub_item_cls == TypeCls.LINE:
                # cast to unicode          
                append_item_to_topic(topic, sub_item, "lines")
            elif sub_item_cls == TypeCls.ANSWER:
                append_item_to_topic(topic, sub_item, "answers")
            else:
                raise Exception("unknown type cls " + str(sub_item_cls))
        if len(topics) == 0:
            dangle_items.append(sub_item)
    # ignore topics without pics and lines
    filter_topic_items = []
    for topic in topic_items:
        if "pictures" in topic or "lines" in topic:
            filter_topic_items.append(topic)
    return filter_topic_items, dangle_items

def find_empty_line_topic(raw_data):
    topic_items = []
    line_items = []
    for item in raw_data:
        cls = item["cls"]
        if cls in TypeCls.topic_clses:
            topic_items.append(item)
        elif cls == TypeCls.LINE:
            line_items.append(item)
            
    for line_item in line_items:
        topics = find_topic_item(topic_items, line_item, 0.6)
        if len(topics) > 0:
            for topic in topics:
                topic_items.remove(topic)
            if len(topic_items) == 0:
                return []
    return topic_items
