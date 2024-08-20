import re
from word import format_converter
from word import rotated_rect_utils

class TopicBuilder:

    def __init__(self, subject):
        self.subject = subject
        self.patterns = [r'第([一二三四五六七八九十]+)部分|Part([ 0-9]+)|(笔试|听力)部分', r'(阅读|短文|文言文|根据[\S]+填空)',
                         r'[\(]?([一二三四五六七八九十]+)[.、 \)]+|([IVXⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)[.、]+',
                         r'([0-9]+)[.、 ]+', r'[\(]?([0-9]+)\)', r'\([ ]*\)([0-9]+)']

    def get_items(self, regions, min_x, max_x):
        items = []
        for item in regions:
            if min_x <= item["region"][0] <= max_x and min_x <= item["region"][2] <= max_x:
                items.append(item)
            elif item["region"][0] <= min_x and item["region"][2] >= max_x:
                if (min_x - item["region"][0]) + (item["region"][2] - max_x) < (
                        item["region"][2] - item["region"][0]) * 0.2:
                    items.append(item)
            elif item["region"][0] <= min_x and min_x < item["region"][2] < max_x:
                if (min_x - item["region"][0]) < (item["region"][2] - item["region"][0]) * 0.2:
                    items.append(item)
            elif min_x < item["region"][0] < max_x and item["region"][2] >= max_x:
                if (item["region"][2] - max_x) < (item["region"][2] - item["region"][0]) * 0.2:
                    items.append(item)
        return items

    def split(self, ret, item=True):
        topics = [item for item in ret["regions"] if 4 <= item["cls"] <= 9]
        if len(topics) == 0:
            return []
        topics.sort(key=lambda x: x["region"][1])
        ts_list = [[] for i in range(len(topics))]
        for i in range(len(topics)-1):
            for j in range(i+1, len(topics)):
                if topics[j]["region"][0] > topics[i]["region"][2] or topics[j]["region"][2] < topics[i]["region"][0]:
                    ts_list[i].append(j)
                    ts_list[j].append(i)
        if all([len(ts) == 0 for ts in ts_list]):
            if item:
                min_x = min([topic["region"][0] for topic in topics])
                max_x = max([topic["region"][2] for topic in topics])
                items = self.get_items(ret["regions"], min_x, max_x)
                return [{"regions": items}]
            else:
                return []
        used = [False for i in range(len(topics))]
        columns = []
        for i in range(len(topics)):
            if used[i]:
                continue
            column = [i]
            used[i] = True
            for j in range(i+1, len(topics)):
                if used[j]:
                    continue
                if ts_list[j] == ts_list[i]:
                    column.append(j)
                    used[j] = True
            columns.append(column)
        rets = []
        for column in columns:
            min_x = min([topics[k]["region"][0] for k in column])
            max_x = max([topics[k]["region"][2] for k in column])
            if item:
                items = self.get_items(ret["regions"], min_x, max_x)
                rets.append({"regions": items})
            else:
                rets.append((min_x, max_x))
        return rets

    def find_pinyin(self, lines):
        pinyin = []
        for i, line in enumerate(lines[:-1]):
            if "result" not in line:
                continue
            result = line["result"][0]
            ret1 = re.findall(r'([a-z ]+)', result)
            if len(ret1) == 0:
                continue
            ret2 = re.findall(r'([āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜ]+)', result)
            if len(ret2) == 0:
                continue
            count1 = sum([len(r) for r in ret1])
            count2 = sum([len(r) for r in ret2])
            if count1 + count2 == len(result):
                if lines[i + 1]["box"][1] - line["box"][3] > line["box"][3] - line["box"][1]:
                    continue
                if len(pinyin) > 0 and pinyin[-1] + 1 == i:
                    pinyin.pop()
                pinyin.append(i)
        return pinyin

    def merge_lines(self, lines, data):
        lines.sort(key=lambda x: x["box"][1])
        lines_list = [[lines[0]]]
        for line in lines[1:]:
            for l in lines_list[-1]:
                center_y = (line["box"][1] + line["box"][3]) / 2
                if l["box"][1] < center_y < l["box"][3]:
                    lines_list[-1].append(line)
                    line = None
                    break
            if line:
                lines_list.append([line])
        for ls in lines_list:
            if len(ls) < 2:
                continue
            ls.sort(key=lambda x: x["box"][0])
            result = ""
            topic = None
            for l in ls:
                lines.remove(l)
                if "result" in l:
                    result += l["result"][0]
                if topic is None:
                    for t in data["topics"]:
                        if "lines" not in t:
                            continue
                        if l in t["lines"]:
                            topic = t
                            topic["lines"].remove(l)
                            break
                elif l in topic["lines"]:
                    topic["lines"].remove(l)
            box = [min([x["box"][0] for x in ls]), min([x["box"][1] for x in ls]),
                   max([x["box"][2] for x in ls]), max([x["box"][3] for x in ls])]
            line = {"cls": 1, "region": box, "box": box, "rotation": 0, "result": [result]}
            lines.append(line)
            if topic:
                topic["lines"].append(line)
                topic["lines"].sort(key=lambda x: x["region"][1])

        lines.sort(key=lambda x: x["box"][1])
        if self.subject == "chinese":
            pinyin = self.find_pinyin(lines)
            k = 0
            for i in pinyin:
                j = i - k
                topic = None
                for t in data["topics"]:
                    if "lines" not in t:
                        continue
                    if lines[j+1] in t["lines"]:
                        topic = t
                        topic["lines"].remove(lines[j+1])
                        break
                lines[j+1]["box"] = [min([lines[j]["box"][0], lines[j+1]["box"][0]]),
                                     min([lines[j]["box"][1], lines[j+1]["box"][1]]),
                                     max([lines[j]["box"][2], lines[j+1]["box"][2]]),
                                     max([lines[j]["box"][3], lines[j+1]["box"][3]])]
                if topic:
                    topic["lines"].append(lines[j+1])
                    topic["lines"].sort(key=lambda x: x["region"][1])
                lines.pop(j)
                k += 1

    def get_line(self, item):
        if "type" in item:
            if "items" in item:
                line = item["items"][0]["lines"][0]
            else:
                line = item["lines"][0]
        else:
            line = item
        return line

    def get_type(self, item, type):
        line = self.get_line(item)
        if "result" in line and len(line["result"][0]) > 0:
            text = line["result"][0]
            if self.subject == "english":
                if re.search(r'(阅读|短文|照样子|完形填空|听录音|根据所听|读一读)', text):
                    type = 3
            elif re.search(r'(阅读|短文|文言文|照样子|完形填空)', text) and '填写' not in text:
                type = 3
        return type

    def construct_topics(self, data):
        regions = data["regions"]
        regions.sort(key=lambda x: x["region"][1])
        topics, dangle_items = format_converter.to_topic_json_format(regions, throw_when_no_topic_found=False)
        data["topics"] = topics
        data["dangle_items"] = dangle_items
        return data

    def construct_topics_v2(self, items, box, pattern):
        titles = []
        for k, item in enumerate(items):
            line = self.get_line(item)
            if "result" not in line or len(line["result"][0]) == 0:
                continue
            m = re.match(pattern, line["result"][0])
            if m:
                titles.append((k, m.group(1)))
        new_topics = []
        ranges = []
        if len(titles) > 0:
            titles.append((len(items), None))
            k = titles[0][0]
            for title in titles[1:]:
                i = title[0]
                items2 = items[k:i]
                type = self.get_type(items2[0], 2)
                if type == 4:
                    for item in items2:
                        item["box"][2] = max(item["box"][2], box[2])
                if i - k > 1:
                    box2 = [min([t["box"][0] for t in items2]), min([t["box"][1] for t in items2]),
                           max([t["box"][2] for t in items2]), max([t["box"][3] for t in items2])]
                else:
                    box2 = items2[0]["box"]
                topic = {"box": box2, "items": items2, "type": type}
                new_topics.append(topic)
                ranges.append([k, i])
                k = i
        return new_topics, ranges

    def build_topic_tree(self, items):
        topic_tree = {"items": items}
        box = [min([t["box"][0] for t in items]), min([t["box"][1] for t in items]),
               max([t["box"][2] for t in items]), max([t["box"][3] for t in items])]
        items_list = [{"items": items, "node": topic_tree, "box": box}]
        for pattern in self.patterns:
            new_items_list = []
            for its in items_list:
                new_topics, ranges = self.construct_topics_v2(its["items"], its["box"], pattern)
                if len(new_topics) > 0:
                    if ranges[0][0] > 0 and its["node"] == topic_tree:
                        sub_items = its["items"][:ranges[0][0]]
                        box = [min([t["box"][0] for t in sub_items]), min([t["box"][1] for t in sub_items]),
                               max([t["box"][2] for t in sub_items]), max([t["box"][3] for t in sub_items])]
                        new_items_list.append({"items": sub_items, "node": topic_tree, "box": box})
                    k = 0
                    for topic, r in zip(new_topics, ranges):
                        r[0] -= k
                        r[1] -= k
                        k += r[1] - r[0] - 1
                        del its["node"]["items"][r[0]:r[1]]
                        t = {"items": topic["items"], "box": topic["box"], "type": topic["type"]}
                        its["node"]["items"].insert(r[0], t)
                        new_items_list.append({"items": topic["items"], "node": t, "box": topic["box"]})
                else:
                    new_items_list.append(its)
            items_list = new_items_list
        for items in items_list:
            for item in items["items"]:
                if item["type"] != 1:
                    continue
                item["type"] = self.get_type(item, item["type"])
                if "pictures" in item:
                    if self.subject == 'english':
                        continue
                    count = 0
                    for picture in item["pictures"]:
                        count = 0
                        for it in item["items"]:
                            line = it["lines"][0]
                            center_y = (line["box"][1] + line["box"][3]) / 2
                            if picture["box"][1] < center_y < picture["box"][3]:
                                count += 1
                        if count > 2:
                            break
                    if count > 2:
                        continue
                items_list = [{"items": item["items"], "node": item, "box": item["box"]}]
                for pattern in self.patterns:
                    new_items_list = []
                    for its in items_list:
                        new_topics, ranges = self.construct_topics_v2(its["items"], its["box"], pattern)
                        if len(new_topics) > 1:
                            if ranges[0][0] > 0 and its["node"] == item:
                                sub_items = its["items"][:ranges[0][0]]
                                box = [min([t["box"][0] for t in sub_items]), min([t["box"][1] for t in sub_items]),
                                       max([t["box"][2] for t in sub_items]), max([t["box"][3] for t in sub_items])]
                                new_items_list.append({"items": sub_items, "node": item, "box": box})
                            k = 0
                            for topic, r in zip(new_topics, ranges):
                                r[0] -= k
                                r[1] -= k
                                k += r[1] - r[0] - 1
                                del its["node"]["items"][r[0]:r[1]]
                                t = {"items": topic["items"], "box": topic["box"], "type": topic["type"]}
                                its["node"]["items"].insert(r[0], t)
                                new_items_list.append({"items": topic["items"], "node": t, "box": topic["box"]})
                        else:
                            new_items_list.append(its)
                    items_list = new_items_list
        return topic_tree

    def get_topic_from_tree(self, tree, regions):
        for item in tree["items"]:
            if item["type"] == 0:
                continue
            if item["type"] == 3:
                regions.append({"cls": 4, "region": item["box"], "rotation": 0})
            else:
                if "items" in item:
                    k = sum(["items" in t for t in item["items"]])
                    if k > 0:
                        self.get_topic_from_tree(item, regions)
                    else:
                        k = sum([t["type"] > 0 for t in item["items"]])
                        if (k == 0) or (item["type"] == 1 and k < 2):
                            regions.append({"cls": 4, "region": item["box"], "rotation": 0})
                        else:
                            self.get_topic_from_tree(item, regions)
                else:
                    regions.append({"cls": 4, "region": item["box"], "rotation": 0})

    def build(self, ret):
        regions = []

        lines = []
        for item in ret["regions"]:
            if item["cls"] != 1:
                continue
            box = item["region"] + [item["rotation"]]
            box = rotated_rect_utils.lefttop_reightbottom_theta_bound_box(box, True)
            if (box[3] - box[1]) > (box[2] - box[0]) * 2:
                continue
            item["box"] = box
            lines.append(item)

        if len(lines) == 0:
            return regions

        str = "一二三四五六七八九十123456789"
        patterns = [r'([一二三四五六七八九十]+)', r'([123456789]+)']
        lines.sort(key=lambda x: x["box"][1])
        for pattern in patterns:
            results = []
            for line in lines:
                if "result" not in line or len(line["result"][0]) < 2:
                    continue
                m = re.match(pattern, line["result"][0])
                if m and len(m.group(1)) == 1:
                    results.append((line, m.group(1)))
            if len(results) > 1:
                k = str.index(results[0][1])
                for result in results[1:]:
                    if str.index(result[1]) == k + 1:
                        k += 1
                    else:
                        k = -1
                        break
                if k > 0:
                    for line, _ in results:
                        s = line["result"][0]
                        line["result"][0] = s[0] + " " + s[1:]
                break

        ret = self.construct_topics(ret)

        self.merge_lines(lines, ret)

        if "result" in lines[-1].keys():
            if re.search(r'第[一二三四五六七八九十0-9]+[页頁真]|[0-9]{1,3}$|共[0-9]+页', lines[-1]["result"][0]):
                del lines[-1]

        items = []
        k = 0
        for topic in ret["topics"]:
            if "lines" not in topic:
                continue
            sub_items = []
            for line in topic["lines"]:
                if line in lines:
                    sub_items.append({"lines": [line], "box": line["box"], "type": 0})
            if len(sub_items) == 0:
                continue
            i = lines.index(sub_items[0]["lines"][0])
            if i > k:
                for j in range(k, i):
                    items.append({"lines": [lines[j]], "type": 0})
            items.append({"region": topic["region"], "rotation": topic["rotation"],
                           "items": sub_items, "type": 1})
            if "pictures" in topic:
                for picture in topic["pictures"]:
                    box = picture["region"] + [picture["rotation"]]
                    box = rotated_rect_utils.lefttop_reightbottom_theta_bound_box(box, True)
                    picture["box"] = box
                items[-1]["pictures"] = topic["pictures"]
            k = lines.index(sub_items[-1]["lines"][0]) + 1
        if k < len(lines):
            for j in range(k, len(lines)):
                items.append({"lines": [lines[j]], "type": 0})

        for item in items:
            if "region" in item:
                box = item["region"] + [item["rotation"]]
                box = rotated_rect_utils.lefttop_reightbottom_theta_bound_box(box, True)
                item["box"] = box
            else:
                item["box"] = item["lines"][0]["box"]
        if len(items) == 0:
            return regions
        topic_tree = self.build_topic_tree(items)
        self.get_topic_from_tree(topic_tree, regions)

        return regions