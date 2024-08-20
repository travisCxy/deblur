using System;
using System.Collections.Generic;
using System.Runtime.Serialization;



namespace imageToWord
{
    [DataContract]
    public class JsonImagesInfo
    {
        [DataMember(Order = 1, Name = "images_info")]
        public JsonData[] jsonDatas;
    }

    [DataContract]
    public class JsonData
    {
        [DataMember(Order = 1, Name = "doc_page_info")]
        public JsonPageInfo jsonPageInfo;
        [DataMember(Order = 2, Name = "doc_textboxs")]
        public JsonTextBox[] jsonTextBoxes;
        [DataMember(Order = 3, Name = "doc_pics")]
        public JsonPic[] jsonPics;
        [DataMember(Order = 4, Name = "doc_sections")]
        public JsonSection[] jsonSections;
        [DataMember(Order = 5, Name = "doc_tables")]
        public JsonTable[] jsonTables;

    }
    [DataContract]
    public class JsonSection
    {
        [DataMember(Order = 1, Name = "doc_colums")]
        public JsonColum[] jsonColums;
    }

        [DataContract]
    public class JsonPageInfo
    {
        [DataMember(Order = 1, Name = "page_top_bottom_margin")]
        public float pageTopBottomMargin;
        [DataMember(Order = 2, Name = "page_left_right_margin")]
        public float pageLeftRightMargin;
        [DataMember(Order = 3, Name = "page_width")]
        public float pageWidth;
        [DataMember(Order = 4, Name = "page_height")]
        public float pageHeight;
        [DataMember(Order = 5, Name = "create_doc")]
        public int createDoc;
    }

    [DataContract]
    public class JsonTextBox
    {
        [DataMember(Order = 1, Name = "page_bbox")]
        public float[] bbox;
        [DataMember(Order = 2, Name = "text")]
        public string text;
        [DataMember(Order = 3, Name = "font_size")]
        public int fontSize;
        [DataMember(Order = 4, Name = "index")]
        public int index;
        [DataMember(Order = 5, Name = "position_h_from_paragraph")]
        public float position_h_from_paragraph1;
        [DataMember(Order = 6, Name = "position_w_from_colum")]
        public float position_w_from_colum;
        [DataMember(Order = 7, Name = "in_page")]
        public int in_page;

        public float position_h_from_paragraph { 
            get { return position_h_from_paragraph1; }
            set { position_h_from_paragraph1 = value; }
        } 
    }

    [DataContract]
    public class JsonPic
    {
        [DataMember(Order = 1, Name = "page_bbox")]
        public float[] bbox;
        [DataMember(Order = 2, Name = "url")]
        public string url;
        [DataMember(Order = 3, Name = "index")]
        public int index;
        [DataMember(Order = 4, Name = "position_h_from_paragraph")]
        public float position_h_from_paragraph1;
        [DataMember(Order = 5, Name = "position_w_from_colum")]
        public float position_w_from_colum;
        [DataMember(Order =6, Name = "in_page")]
        public int in_page;
        [DataMember(Order = 7, Name = "pad_text")]
        public int pad_text;
        [DataMember(Order = 8, Name = "warp")]
        public string warp;
        public float position_h_from_paragraph
        {
            get { return position_h_from_paragraph1; }
            set { position_h_from_paragraph1 = value; }
        }

    }

    [DataContract]
    public class JsonColum
    {
        [DataMember(Order = 1, Name = "page_range")]
        public float[] pageRange;
        [DataMember(Order = 2, Name = "colum_space")]
        public float columSpace;
        [DataMember(Order = 3, Name = "paragraphs")]
        public JsonParagraph[] jsonParagraphs;
        [DataMember(Order = 4, Name = "blocks")]
        public JsonBlock[] jsonBlocks;
    }

    [DataContract]
    public class JsonBlock
    {
        [DataMember(Order = 1, Name = "warp")]
        public string warp;
        [DataMember(Order = 2, Name = "paragraphs")]
        public JsonParagraph[] jsonParagraphs;
        [DataMember(Order = 3, Name = "pics")]
        public JsonPic[] jsonPics;
        [DataMember(Order = 4, Name = "tables")]
        public JsonTable[] jsonTables;
        [DataMember(Order = 5, Name = "keep_para")]
        public int keepPara;
        [DataMember(Order = 6, Name = "create_para")]
        public int createPara;
        [DataMember(Order = 7, Name = "n_pad_line")]
        public int pad_line;
    }

    [DataContract]
    public class JsonParagraph
    {
        [DataMember(Order = 1, Name = "font_size")]
        public int fontSize;
        [DataMember(Order = 2, Name = "line_space")]
        public float lineSpace;
        [DataMember(Order = 3, Name = "line_space_rule")]
        public string line_space_rule;
        [DataMember(Order = 3, Name = "n_pad_line")]
        public int padLine;
        [DataMember(Order = 4, Name = "para_space_before")]
        public float spaceBefore;
        [DataMember(Order = 5, Name = "align")]
        public string align;
        [DataMember(Order = 6, Name = "lines")]
        public JsonLine[] lines;
        [DataMember(Order = 7, Name = "right_indent")]
        public float rightIndent;
        [DataMember(Order = 8, Name = "float_content_id")]
        public int[] float_content_id;
        [DataMember(Order = 9, Name = "page_height_start")]
        public float page_height_start;
        [DataMember(Order = 10, Name = "page_height_end")]
        public float page_height_end;
        [DataMember(Order = 11, Name = "black")]
        public int blackFont;
        [DataMember(Order = 12, Name = "cell_before_pics")]
        public int[] cell_before_pics;
        [DataMember(Order = 13, Name = "cell_follow_pics")]
        public int[] cell_follow_pics;
        public float leftIndent
        {
            get {
                if (lines.Length < 1 || align == "center")
                {
                    return (float)0;
                }
                else if (lines.Length == 1) {
                    return lines[0].lineIndent;
                }
                else
                {
                    return lines[1].lineIndent;
                }
            }
        }
        public float page_height
        {
            get
            {
                return page_height_end - page_height_start;
            }
        }
        public float firstLineLeftIndent
        {
            get
            {
                if (align == "center" || lines.Length<=1) {
                    return 0;
                }
                return lines[0].lineIndent - leftIndent;
            }
        }
    }

    [DataContract]
    public class JsonLine
    {
        [DataMember(Order = 1, Name = "font_size")]
        public int fontSize;
        [DataMember(Order = 2, Name = "left_indent")]
        public float lineIndent;
        [DataMember(Order = 3, Name = "line_text")]
        public JsonText[] texts;
        [DataMember(Order = 4, Name = "w_ratio")]
        public float wordRatio;
        [DataMember(Order = 5, Name = "w_space")]
        public float wordSpace;
        [DataMember(Order = 6, Name = "use_tab")]
        public int useTab;
        [DataMember(Order = 7, Name = "sentences")]
        public JsonSentence[] sentences;

    }

    [DataContract]
    public class JsonSentence
    {
        
        [DataMember(Order = 1, Name = "text")]
        public string text;
    }


    [DataContract]
    public class JsonText
    {
        [DataMember(Order = 1, Name = "text")]
        public string text;
    }
        [DataContract]
    public class JsonTable
    {
        [DataMember(Order = 1, Name = "page_bbox")]
        public float[] bbox;
        [DataMember(Order = 2, Name = "col_num")]
        public int col_num;
        [DataMember(Order = 3, Name = "row_num")]
        public int row_num;
        [DataMember(Order = 4, Name = "cells")]
        public JsonCell[] cells;
        [DataMember(Order = 5, Name = "row_heights")]
        public float[] row_heights;
        [DataMember(Order = 6, Name = "col_widths")]
        public float[] col_widths;
        [DataMember(Order = 7, Name = "cell_index")]
        public JsonColumIndex[] cell_index;
        [DataMember(Order = 8, Name = "index")]
        public int index;
        [DataMember(Order = 9, Name = "position_h_from_paragraph")]
        public float position_h_from_paragraph1;
        [DataMember(Order = 10, Name = "position_w_from_colum")]
        public float position_w_from_colum;
        [DataMember(Order = 11, Name = "in_page")]
        public int in_page;
        [DataMember(Order = 12, Name = "warp")]
        public string warp;
        public float position_h_from_paragraph
        {
            get { return position_h_from_paragraph1; }
            set { position_h_from_paragraph1 = value; }
        }

        public float allCellHeight
        {
            get {
                float tmp = 0;
                for (int i = 0; i < row_heights.Length; i++)
                {
                    tmp += row_heights[i]; 
                }
                return tmp;
            }
        }
        public float allCellWidth
        {
            get
            {
                float tmp = 0;
                for (int i = 0; i < col_widths.Length; i++)
                {
                    tmp += col_widths[i];
                }
                return tmp;
            }
        }
        public float Width {get{ return bbox[2] - bbox[0]; }}
        public float Height {get{ return bbox[3] - bbox[1]; }}
        public float ScaleW {get{ return Width/allCellWidth ; }}
        public float ScaleH { get{ return Height/allCellHeight; } }

    }

    [DataContract]
    public class JsonCell
    {
        [DataMember(Order = 9, Name = "paragraphs")]
        public JsonParagraph[] paragraphs;
        [DataMember(Order = 10, Name = "cell_pics")]
        public JsonPic[] pics;
        public int fontSize { get
            {
                if (paragraphs.Length == 0) {
                    return 0;
                }
                return paragraphs[0].fontSize;
            } }
    }


    [DataContract]
    public class JsonColumIndex
    {
        [DataMember(Order = 0, Name = "row")]
        public int[] row;
    }
 }