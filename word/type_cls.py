class TypeCls:
    LINE = 1
    ANSWER = 2
    PICTURE = 3
    TOPIC_FILL = 4
    TOPIC_CHOOSE = 5
    TOPIC_YESNO = 6
    TOPIC_CALC = 7
    TOPIC_OPERATE = 8
    TOPIC_ANSWER = 9
    STUDENT = 10

    topic_clses = [TOPIC_FILL, TOPIC_CHOOSE, TOPIC_YESNO, TOPIC_CALC, TOPIC_OPERATE, TOPIC_ANSWER]


    inner_picture = 3
    inner_line = 1
    inner_topic = 4
    inner_handwrite = 2

    @staticmethod
    def convert_inner(inner):
        if inner == 5:
            inner = TypeCls.STUDENT
        if inner == 6:
            inner = TypeCls.ANSWER
        return inner
        #inner = int(inner)
        #if inner == TypeCls.inner_picture:
        #    return TypeCls.PICTURE
        #elif inner == TypeCls.inner_line:
        #    return TypeCls.LINE
        #elif inner == TypeCls.inner_handwrite:
        #    return TypeCls.ANSWER
        #elif inner == TypeCls.inner_topic:
        #    return TypeCls.TOPIC_FILL
        #
        #else:
        #    raise Exception("unknown inner cls" + str(inner))

if __name__ == '__main__':
    from env import raw_data_path
    import codecs
    import os
    import json
    for dirname in ['batch1', 'batch2']:
        
        for fname in os.listdir(os.path.join(raw_data_path, dirname)):
            fname = os.path.join(raw_data_path, dirname, fname)

            if fname.endswith('.txt'):
                old_file = fname + ".old"
                if not os.path.exists(old_file):
                    with codecs.open(fname, 'r', 'utf-8') as f:
                        data = json.load(f)
                    with codecs.open(old_file, 'w', 'utf-8') as f:
                        json.dump(data, f, indent = 4, ensure_ascii = False)

                with codecs.open(old_file, 'r', 'utf-8') as f:
                    data = json.load(f)
                for region in data['regions']:
                    cls = int(region['cls'])
                    if cls == 0:
                        cls = TypeCls.PICTURE
                    elif cls == 1:
                        cls = TypeCls.LINE
                    elif cls == 101:
                        cls = TypeCls.TOPIC_FILL
                    elif cls == 102:
                        cls = TypeCls.ANSWER
                    else:
                        raise Exception('unknown cls: %d' % cls)
                    region['cls'] = cls
                with codecs.open(fname, 'w', 'utf-8') as f:
                    json.dump(data, f, indent = 4, ensure_ascii = False)