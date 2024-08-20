
using System.Text;
using System.Runtime.Serialization.Json;
using System.Drawing.Imaging;
using System.Xml.Linq;
using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;
using Table = DocumentFormat.OpenXml.Wordprocessing.Table;
using TableCell = DocumentFormat.OpenXml.Wordprocessing.TableCell;
using Path = System.IO.Path;

using A = DocumentFormat.OpenXml.Drawing;
using DW = DocumentFormat.OpenXml.Drawing.Wordprocessing;
using PIC = DocumentFormat.OpenXml.Drawing.Pictures;

namespace imageToWord
{

    class WordDocBuilder
    {
        public static int floatContentParaIndex = 0;
        public static int shape_id = 1;
        public static T ReadJson<T>(string path, DataContractJsonSerializer ser) where T : class
        {
            using (Stream stream = File.OpenRead(path))
            {
                T json = ser.ReadObject(stream) as T;
                return json;
            }
        }
        public static void CreateDoc1(string jsonPath, string outDocPath)
        {
            JsonImagesInfo meta_datas = ReadJson<JsonImagesInfo>(jsonPath, new DataContractJsonSerializer(typeof(JsonImagesInfo)));
            using (WordprocessingDocument doc = WordprocessingDocument.Create(outDocPath, WordprocessingDocumentType.Document))
            {
                //全局初始化
                 MainDocumentPart mainPart = doc.AddMainDocumentPart();
                StyleDefinitionsPart part = DocUtils.AddStylesPartToPackage(doc);
                DocUtils.CreateAndAddCharacterStyle(part);
                mainPart.Document = DocUtils.GetInitDocument();
                DocUtils.AddSettings(mainPart);
                Body body = mainPart.Document.AppendChild(new Body());

                for (int image_id = 0; image_id < meta_datas.jsonDatas.Length; image_id++)
                {
                    JsonData meta_data = meta_datas.jsonDatas[image_id];
                    floatContentParaIndex = body.Elements<Paragraph>().Count() + 1;
                    //按节添加本页内容(文本)
                    for (int sectId = 0; sectId < meta_data.jsonSections.Length; sectId++)
                    {
                        bool nextPage = false;
                        //添加内容块
                        JsonColum[] colums = meta_data.jsonSections[sectId].jsonColums;
                        if (colums.Length > 1)
                        {
                            for (int colum_id = 0; colum_id < colums.Length; colum_id++)
                            {
                                AddColumBlocks(body, colums[colum_id], mainPart);
                                if (colum_id != colums.Length - 1)
                                {
                                    body.Append(new Paragraph(new Run(new Break() { Type = BreakValues.Page })));
                                }
                            }
                        }
                        else
                        {
                            for (int colum_id = 0; colum_id < colums.Length; colum_id++)
                            {
                                bool addBreak = false;
                                if (colums.Length >= 2 && colum_id != colums.Length - 1)
                                {
                                    addBreak = true;
                                }
                                AddColum(body, colums[colum_id], addBreak, meta_data, mainPart);
                            }
                        }

                        AddSection(body, meta_data.jsonPageInfo, meta_data.jsonSections[sectId], nextPage);
                        //最后一个sectPr不需要进行移动
                        if (image_id != meta_datas.jsonDatas.Length - 1 || sectId != meta_data.jsonSections.Length - 1)
                        {
                            DocUtils.MoveSectionPr(body);
                        }
                    }

                    if (body.Elements<Paragraph>().Count() == 0)
                    {
                        body.Append(new Paragraph());
                    }

                    //添加文本框
                    JsonTextBox[] textBoxes = meta_data.jsonTextBoxes;
                    AddTextBoxes(body, textBoxes);
                    //添加表格
                    JsonTable[] tables = meta_data.jsonTables;
                    AddTableInTextBox(mainPart, body, tables);
                    //添加图片
                    JsonPic[] pictures = meta_data.jsonPics;
                    AddFloatPictures(mainPart, body, pictures);
                    IEnumerable<Paragraph> allParagraphs = body.Elements<Paragraph>();
                    Paragraph last_p = allParagraphs.Last();
                    //增加分页符号
                    if (image_id != meta_datas.jsonDatas.Length - 1)
                    {
                        Run r = last_p.AppendChild(new Run(new Break() { Type = BreakValues.Page }));
                    }
                }
            }
        }

        public static void CreateDocFromTemplateDocx(string jsonPath, string outDocPath)
        {
            JsonImagesInfo meta_datas = ReadJson<JsonImagesInfo>(jsonPath, new DataContractJsonSerializer(typeof(JsonImagesInfo)));
            using (WordprocessingDocument doc = WordprocessingDocument.Open(outDocPath, true))
            {
                //全局初始化
                MainDocumentPart mainPart = doc.MainDocumentPart;
                //MainDocumentPart mainPart = doc.AddMainDocumentPart();
                Body body = mainPart.Document.AppendChild(new Body());

                for (int image_id = 0; image_id < meta_datas.jsonDatas.Length; image_id++)
                {
                    JsonData meta_data = meta_datas.jsonDatas[image_id];
                    floatContentParaIndex = body.Elements<Paragraph>().Count() + 1;
                    //按节添加本页内容(文本)
                    for (int sectId = 0; sectId < meta_data.jsonSections.Length; sectId++)
                    {
                        bool nextPage = false;
                        //添加内容块
                        JsonColum[] colums = meta_data.jsonSections[sectId].jsonColums;
                        for (int colum_id = 0; colum_id < colums.Length; colum_id++)
                        {
                            bool addBreak = false;
                            if (colums.Length >= 2 && colum_id != colums.Length - 1)
                            //if (colums.Length >= 2 && colum_id != 0)
                            {
                                addBreak = true;
                            }
                            AddColum(body, colums[colum_id], addBreak, meta_data, mainPart);
                        }
                        AddSection(body, meta_data.jsonPageInfo, meta_data.jsonSections[sectId], nextPage);
                        //最后一个sectPr不需要进行移动
                        if (image_id != meta_datas.jsonDatas.Length - 1 || sectId != meta_data.jsonSections.Length - 1)
                        {
                            DocUtils.MoveSectionPr(body);
                        }
                    }

                    if (body.Elements<Paragraph>().Count() == 0)
                    {
                        body.Append(new Paragraph());
                    }

                    //添加文本框
                    JsonTextBox[] textBoxes = meta_data.jsonTextBoxes;
                    AddTextBoxes(body, textBoxes);
                    //添加表格
                    JsonTable[] tables = meta_data.jsonTables;
                    AddTableInTextBox(mainPart, body, tables);
                    //添加图片
                    JsonPic[] pictures = meta_data.jsonPics;
                    AddFloatPictures(mainPart, body, pictures);
                    IEnumerable<Paragraph> allParagraphs = body.Elements<Paragraph>();
                    Paragraph last_p = allParagraphs.Last();
                    //增加分页符号
                    if (image_id != meta_datas.jsonDatas.Length - 1)
                    {
                        Run r = last_p.AppendChild(new Run(new Break() { Type = BreakValues.Page }));
                    }
                }
            }
        }
        public static void AddSection(Body body, JsonPageInfo jsonPageInfo, JsonSection jsonSection, bool nextPage)
        {
            float page_pad = (float)0.2;
            JsonColum[] colums = jsonSection.jsonColums;
            int nColumCount = colums.Length;
            page_pad = nColumCount * page_pad;
            //添加新的全局sectPr
            SectionProperties sectionProperty = new SectionProperties();
            //类型
            SectionType secType = new SectionType();
            if (nextPage)
            {
                secType.Val = SectionMarkValues.NextPage;
            }
            else
            {
                secType.Val = SectionMarkValues.Continuous;
            }
            sectionProperty.Append(secType);

            //设置页面大小
            PageSize pgSz = new PageSize();
            pgSz.Width = Convert.ToUInt32(jsonPageInfo.pageWidth * 567);
            pgSz.Height = Convert.ToUInt32(jsonPageInfo.pageHeight * 567);
            //pgSz.Orient = PageOrientationValues.Landscape;

            //页边距
            PageMargin pgMr = new PageMargin(); ;
            pgMr.Left = Convert.ToUInt32((jsonPageInfo.pageLeftRightMargin- page_pad) * 567);
            pgMr.Right = Convert.ToUInt32((jsonPageInfo.pageLeftRightMargin- page_pad) * 567);
            pgMr.Top = Convert.ToInt32(jsonPageInfo.pageTopBottomMargin * 567);
            pgMr.Bottom = Convert.ToInt32(jsonPageInfo.pageTopBottomMargin * 567);
            pgMr.Header = 0;
            pgMr.Footer = 0;

            DocGrid docGrid = new DocGrid();
            //docGrid.LinePitch = Convert.ToInt32(312);
            docGrid.LinePitch = Convert.ToInt32(312);
            docGrid.Type = DocGridValues.Lines;

            //栏属性
            Columns pColums = new Columns();

            //if (colums.Length <= 1)
            //{
            //    pColums.ColumnCount = Convert.ToInt16(1);
            //}
            //else
            //{
            //    pColums.ColumnCount = Convert.ToInt16(colums.Length);
            //    //pColums.Space = (Convert.ToUInt32(Math.Max(colums[0].columSpace * 567, 0))).ToString();
            //    pColums.Space = (Convert.ToUInt32(Math.Max(567, 0))).ToString();
            //    pColums.EqualWidth = true;
            //    float colum_pad = 2 * page_pad / colums.Length;
            //    for (int colId = 0; colId < colums.Length; colId++)
            //    {
            //        float colWidth = colums[colId].pageRange[2] - colums[colId].pageRange[0] + colum_pad;
            //        float colSpace = colums[colId].columSpace;
            //        Column column = new Column();
            //        column.Width = (Convert.ToUInt32(colWidth * 567)).ToString();
            //        column.Space = (Convert.ToUInt32(Math.Max(colSpace * 567, 0))).ToString();
            //        pColums.AppendChild(column);
            //    }
            //}
            sectionProperty.Append(pgSz);
            sectionProperty.Append(pgMr);
            sectionProperty.Append(pColums);
            sectionProperty.Append(docGrid);
            body.AppendChild<SectionProperties>(sectionProperty);
        }


        public static void AddColum(Body body, JsonColum colum_meta_data, bool addBreak, JsonData meta_data, MainDocumentPart mainPart)
        {
            bool paraAddBreak = false;
            for (int id = 0; id < colum_meta_data.jsonParagraphs.Length; id++)
            {

                if (id == colum_meta_data.jsonParagraphs.Length - 1 && addBreak)
                //if (id == 0 && addBreak)
                {
                    paraAddBreak = true;
                }
                AddParagraph(body, colum_meta_data.jsonParagraphs[id], paraAddBreak, meta_data, mainPart);
                paraAddBreak = false;
            }
        }

        public static void AddColumBlocks(Body body, JsonColum colum_meta_data, MainDocumentPart mainPart)
        {

            Paragraph last_p = null;
            for (int block_id = 0; block_id < colum_meta_data.jsonBlocks.Length; block_id++)
            {
                JsonParagraph[] paras = colum_meta_data.jsonBlocks[block_id].jsonParagraphs;
                JsonTable[] tables = colum_meta_data.jsonBlocks[block_id].jsonTables;
                JsonPic[] pics = colum_meta_data.jsonBlocks[block_id].jsonPics;
                string block_align = colum_meta_data.jsonBlocks[block_id].warp;
                int keepPara = colum_meta_data.jsonBlocks[block_id].keepPara;
                int createPara = colum_meta_data.jsonBlocks[block_id].createPara;
                int n_pad_line = colum_meta_data.jsonBlocks[block_id].pad_line;
                if (paras.Length != 0) {
                    for (int pid = 0; pid < paras.Length; pid++) {
                        JsonParagraph para_meta_data = paras[pid];
                        int blackFont = para_meta_data.blackFont;
                        float spaceBefore = para_meta_data.spaceBefore;
                        float lineSpace = para_meta_data.lineSpace;
                        float firstLineLeftIndent = para_meta_data.firstLineLeftIndent;
                        float lineIndent = para_meta_data.leftIndent;
                        float rightIndent = Math.Max(0, para_meta_data.rightIndent - (float)0.20);
                        string align = para_meta_data.align;
                        string line_space_rule = para_meta_data.line_space_rule;
                        Paragraph p = new Paragraph();
                        ParagraphProperties pPr = new ParagraphProperties();
                        int fontSize = para_meta_data.fontSize;
                        float fontSizeCm = fontSize * (float)0.03528;
                        //间距
                        SpacingBetweenLines space = new SpacingBetweenLines();
                        space.Before = (Convert.ToUInt32(spaceBefore * 567)).ToString();
                        if (line_space_rule == "exact")
                        {
                            space.Line = (Convert.ToUInt32(Math.Max(lineSpace, 0) * 567)).ToString();
                            space.LineRule = LineSpacingRuleValues.Exact;
                        }
                        else if (line_space_rule == "onehalf")
                        {
                            space.Line = 300.ToString();
                            space.LineRule = LineSpacingRuleValues.Auto;
                        }
                        //缩进
                        Indentation indentation = new Indentation();
                        indentation.Left = (Convert.ToUInt32(lineIndent * 567)).ToString();
                        if (firstLineLeftIndent > 0)
                        {
                            indentation.FirstLine = (Convert.ToUInt32(firstLineLeftIndent * 567)).ToString();
                        }
                        else
                        {
                            indentation.Hanging = (Convert.ToUInt32(Math.Abs(firstLineLeftIndent) * 567)).ToString();
                        }
                        if (para_meta_data.lines.Length > 1)
                        {
                            indentation.Right = (Convert.ToUInt32(rightIndent * 567)).ToString();
                        }

                        //中西文间距
                        AutoSpaceDE autoSpaceDE = new AutoSpaceDE() { Val = false };
                        AutoSpaceDN autoSpaceDN = new AutoSpaceDN() { Val = false };

                        //对齐
                        Justification justification = new Justification() { Val = JustificationValues.Both };
                        if (para_meta_data.lines.Length == 1 || align == "left")
                        {
                            justification.Val = JustificationValues.Left;
                        }
                        if (align == "center")
                        {
                            justification.Val = JustificationValues.Center;
                        }
                        TextAlignment alignment = new TextAlignment() { Val = VerticalTextAlignmentValues.Baseline };

                        //Italic italic = new Italic() { };
                        //设置段落属性
                        pPr.Append(space);
                        pPr.Append(indentation);
                        pPr.Append(autoSpaceDE);
                        pPr.Append(autoSpaceDN);
                        pPr.Append(justification);
                        pPr.Append(alignment);
                        //pPr.Append(italic);
                        p.Append(pPr);

                        for (int id = 0; id < para_meta_data.lines.Length; id++)
                        {
                            JsonLine line = para_meta_data.lines[id];
                            float wordRatio = line.wordRatio;
                            float wordSpace = line.wordSpace;
                            JsonText[] line_texts = line.texts;
                            int useTab = line.useTab;
                            for (int text_id = 0; text_id < line_texts.Length; text_id++)
                            {
                                string text = line_texts[text_id].text;
                                if (text == "")
                                {
                                    continue;
                                }
                                bool isOmml = text.Contains("omml");
                                if (isOmml)
                                {
                                    string ommlStr = text.Split("omml")[1];
                                    DocUtils.AddMathFormula(p, ommlStr, fontSize);
                                    //alignment.Val = VerticalTextAlignmentValues.Baseline;
                                    alignment.Val = VerticalTextAlignmentValues.Center;
                                    continue;
                                }
                                Run run = p.AppendChild(new Run());
                                RunProperties rPr = DocUtils.GetRunProperties(fontSize, blackFont);
                                rPr = DocUtils.SetWordSpace(rPr, wordRatio, wordSpace);
                                if (text[0] != '^')
                                {
                                    run.AddChild(rPr);
                                    Text lineText = new Text(text);
                                }
                                else if (text.Length >= 2 && text[1] == '^')
                                {
                                    VerticalTextAlignment v = new VerticalTextAlignment() { Val = VerticalPositionValues.Subscript };
                                    rPr.Append(v);
                                    run.AddChild(rPr);
                                    string new_text = "";
                                    for (int j = 2; j < text.Length; j++)
                                    {
                                        new_text += text[j];
                                    }
                                    text = new_text;
                                }
                                else
                                {
                                    string new_text = "";
                                    for (int j = 1; j < text.Length; j++)
                                    {
                                        new_text += text[j];
                                    }
                                    VerticalTextAlignment v = new VerticalTextAlignment() { Val = VerticalPositionValues.Superscript };
                                    rPr.Append(v);
                                    run.AddChild(rPr);
                                    text = new_text;
                                }
                                Text t = new Text(text);
                                t.Space = SpaceProcessingModeValues.Preserve;
                                run.Append(t);
                            }
                        }
                        if (keepPara == 1) {
                            last_p = p;
                        }
                        body.Append(p);
                        int pad_line = para_meta_data.padLine;
                        for (int pad_id = 0; pad_id < pad_line; pad_id++)
                        {
                            Paragraph p1 = new Paragraph();
                            ParagraphProperties pPr1 = new ParagraphProperties();
                            float padLineSpace = (float)0.0352 * fontSize;
                            //间距
                            SpacingBetweenLines space1 = new SpacingBetweenLines();
                            space1.Before = (Convert.ToUInt32(0)).ToString();
                            space1.After = (Convert.ToUInt32(0)).ToString();
                            space1.Line = (Convert.ToUInt32(padLineSpace * 567)).ToString();
                            space1.LineRule = LineSpacingRuleValues.Exact;
                            pPr1.Append(space1);
                            p1.Append(pPr1);
                            body.Append(p1);
                        }
                    }
                }
                if (pics.Length != 0)
                {
                    for (int pid = 0; pid < pics.Length; pid++) {
                        JsonPic pic_data = pics[pid];
                        int pad_text = pic_data.pad_text;
                        string url = pic_data.url;
                        float[] bbox = pic_data.bbox;
                        float position_w_from_colum = pic_data.position_w_from_colum;
                        float position_h_from_paragraph = pic_data.position_h_from_paragraph;
                        List<float> position = new List<float>() { position_w_from_colum, position_h_from_paragraph, bbox[2] - bbox[0], bbox[3] - bbox[1] };
                        

                        ImagePart imagePart = mainPart.AddImagePart(ImagePartType.Jpeg);
                        using (FileStream stream = new FileStream(url, FileMode.Open))
                        //using (FileStream stream = new FileStream("qq.png", FileMode.Open))
                        {
                            imagePart.FeedData(stream);
                        }
                        Run imageRun = new Run();
                        if (block_align == "warpboth")
                        {
                            Drawing pic = DocUtils.GetAnchorPicture(mainPart.GetIdOfPart(imagePart), Path.GetFileNameWithoutExtension(url), position, shape_id, false, true, "both");
                            imageRun.Append(pic);
                            if (createPara == 1 || last_p == null)
                            {
                                body.Append(new Paragraph(imageRun));
                            }
                            else
                            {
                                last_p.Append(imageRun);
                            }

                        }
                        else if (block_align == "warpleft") {
                            Drawing pic = DocUtils.GetAnchorPicture(mainPart.GetIdOfPart(imagePart), Path.GetFileNameWithoutExtension(url), position, shape_id, false, true, "left");
                            imageRun.Append(pic);
                            if (createPara == 1 || last_p == null)
                            {
                                body.Append(new Paragraph(imageRun));
                            }
                            else
                            {
                                last_p.Append(imageRun);
                            }
                        }
                        else if (block_align == "warpright")
                        {
                            Drawing pic = DocUtils.GetAnchorPicture(mainPart.GetIdOfPart(imagePart), Path.GetFileNameWithoutExtension(url), position, shape_id, false, true, "right");
                            imageRun.Append(pic);
                            if (createPara == 1 || last_p == null)
                            {
                                body.Append(new Paragraph(imageRun));
                            }
                            else
                            {
                                last_p.Append(imageRun);
                            }
                        }
                        else
                        {
                            Drawing pic = DocUtils.CreateInlineDrawing(mainPart.GetIdOfPart(imagePart), position, shape_id);
                            imageRun.Append(pic);
                            if (createPara == 1 || last_p==null)
                            {
                                Paragraph new_p = new Paragraph();
                                if (position_w_from_colum <= 2.5)
                                {
                                    new_p.Append(new ParagraphProperties(new Justification() { Val = JustificationValues.Left }));
                                }
                                else if (position_w_from_colum >= 7.5)
                                {
                                    new_p.Append(new ParagraphProperties(new Justification() { Val = JustificationValues.Right }));
                                }
                                else
                                {
                                    new_p.Append(new ParagraphProperties(new Justification() { Val = JustificationValues.Center }));
                                }
                                new_p.Append(imageRun);
                                body.Append(new_p);
                                last_p = new_p;
                            }
                            else
                            {
                                last_p.Append(imageRun);
                            }
                            last_p.Append(DocUtils.GetTabRun());
                        }
                        shape_id += 1;
                        for (int pad_id = 0; pad_id <n_pad_line; pad_id++)
                        {
                            Paragraph p1 = new Paragraph();
                            ParagraphProperties pPr1 = new ParagraphProperties();
                            float padLineSpace = (float)0.0352 * 12;
                            //间距
                            SpacingBetweenLines space1 = new SpacingBetweenLines();
                            space1.Before = (Convert.ToUInt32(0)).ToString();
                            space1.After = (Convert.ToUInt32(0)).ToString();
                            space1.Line = (Convert.ToUInt32(padLineSpace * 567)).ToString();
                            space1.LineRule = LineSpacingRuleValues.Exact;
                            pPr1.Append(space1);
                            p1.Append(pPr1);
                            body.Append(p1);
                        }
                    }
                }

                if (tables.Length != 0) {
                    for (int tid = 0; tid < tables.Length; tid++) {
                        Paragraph new_p = new Paragraph();
                        Table t = AddTableWithInPara(mainPart, new_p, tables[tid], false);
                        body.Append(t);
                    }
                    for (int pad_id = 0; pad_id < n_pad_line; pad_id++)
                    {
                        Paragraph p1 = new Paragraph();
                        ParagraphProperties pPr1 = new ParagraphProperties();
                        float padLineSpace = (float)0.0352 * 12;
                        //间距
                        SpacingBetweenLines space1 = new SpacingBetweenLines();
                        space1.Before = (Convert.ToUInt32(0)).ToString();
                        space1.After = (Convert.ToUInt32(0)).ToString();
                        space1.Line = (Convert.ToUInt32(padLineSpace * 567)).ToString();
                        space1.LineRule = LineSpacingRuleValues.Exact;
                        pPr1.Append(space1);
                        p1.Append(pPr1);
                        body.Append(p1);
                    }
                }
            }
            
        }
        public static void AddParagraph(Body body, JsonParagraph para_meta_data, bool paraAddBreak, JsonData meta_data, MainDocumentPart mainPart)
        {
            float spaceBefore = para_meta_data.spaceBefore;
            float lineSpace = para_meta_data.lineSpace;
            float firstLineLeftIndent = para_meta_data.firstLineLeftIndent;
            float lineIndent = para_meta_data.leftIndent;
            //float rightIndent = Math.Max(0, para_meta_data.rightIndent - (float)0.20);
            float rightIndent = 0;
            string align = para_meta_data.align;
            string line_space_rule = para_meta_data.line_space_rule;
            Paragraph p = new Paragraph();
            ParagraphProperties pPr = new ParagraphProperties();
            int fontSize = para_meta_data.fontSize;
            float fontSizeCm = fontSize * (float)0.03528;
            //间距
            SpacingBetweenLines space = new SpacingBetweenLines();
            space.Before = (Convert.ToUInt32(spaceBefore * 567)).ToString();
            if (line_space_rule == "exact")
            {
                space.Line = (Convert.ToUInt32(Math.Max(lineSpace, 0) * 567)).ToString();
                space.LineRule = LineSpacingRuleValues.Exact;
            }
            else if(line_space_rule == "onehalf"){
                space.Line = 300.ToString();
                space.LineRule = LineSpacingRuleValues.Auto;
            }
            //缩进
            Indentation indentation = new Indentation();
            indentation.Left = (Convert.ToUInt32(lineIndent * 567)).ToString();
            if (firstLineLeftIndent > 0)
            {
                indentation.FirstLine = (Convert.ToUInt32(firstLineLeftIndent * 567)).ToString();
            }
            else
            {
                indentation.Hanging = (Convert.ToUInt32(Math.Abs(firstLineLeftIndent) * 567)).ToString();
            }
            if (para_meta_data.lines.Length > 1)
            {
                indentation.Right = (Convert.ToUInt32(rightIndent * 567)).ToString();
            }

            //中西文间距
            AutoSpaceDE autoSpaceDE = new AutoSpaceDE() { Val = false };
            AutoSpaceDN autoSpaceDN = new AutoSpaceDN() { Val = false };

            //对齐
            Justification justification = new Justification() { Val = JustificationValues.Both };
            if (para_meta_data.lines.Length == 1 || align== "left")
            {
                justification.Val = JustificationValues.Left;
            }
            if (align == "center")
            {
                justification.Val = JustificationValues.Center;
            }
            TextAlignment alignment = new TextAlignment() { Val = VerticalTextAlignmentValues.Baseline };

            //设置段落属性
            pPr.Append(space);
            pPr.Append(indentation);
            pPr.Append(autoSpaceDE);
            pPr.Append(autoSpaceDN);
            pPr.Append(justification);
            pPr.Append(alignment);
            RunProperties runProperties = new RunProperties(new Italic());
            pPr.Append(runProperties);
            p.Append(pPr);


            int blackFont = para_meta_data.blackFont;
            for (int id = 0; id < para_meta_data.lines.Length; id++)
            {
                JsonLine line = para_meta_data.lines[id];
                float wordRatio = line.wordRatio;
                float wordSpace = line.wordSpace;
                JsonText[] line_texts = line.texts;
                for (int text_id = 0; text_id < line_texts.Length; text_id++)
                {
                    string text = line_texts[text_id].text;
                    if (text == "")
                    {
                        continue;
                    }
                    bool isOmml = text.Contains("omml");
                    if (isOmml)
                    {
                        string ommlStr = text.Split("omml")[1];
                        DocUtils.AddMathFormula(p, ommlStr, fontSize);
                        alignment.Val = VerticalTextAlignmentValues.Center;
                        continue;
                    }
                    Run run = p.AppendChild(new Run());
                    RunProperties rPr = DocUtils.GetRunProperties(fontSize, blackFont);
                    rPr = DocUtils.SetWordSpace(rPr, wordRatio, wordSpace);
                    if (text[0] != '^')
                    {
                        run.AddChild(rPr);
                        Text lineText = new Text(text);
                    }
                    else if (text.Length >= 2 && text[1] == '^')
                    {
                        VerticalTextAlignment v = new VerticalTextAlignment() { Val = VerticalPositionValues.Subscript };
                        rPr.Append(v);
                        run.AddChild(rPr);
                        string new_text = "";
                        for (int j = 2; j < text.Length; j++)
                        {
                            new_text += text[j];
                        }
                        text = new_text;
                    }
                    else
                    {
                        string new_text = "";
                        for (int j = 1; j < text.Length; j++)
                        {
                            new_text += text[j];
                        }
                        VerticalTextAlignment v = new VerticalTextAlignment() { Val = VerticalPositionValues.Superscript };
                        rPr.Append(v);
                        run.AddChild(rPr);
                        text = new_text;
                    }
                    Text t = new Text(text);
                    t.Space = SpaceProcessingModeValues.Preserve;
                    run.Append(t);
                }
            }

            int[] float_content_id = para_meta_data.float_content_id;
            body.Append(p);
            if (float_content_id.Length != 0)
            {

                Paragraph pp = new Paragraph(new ParagraphProperties(new SpacingBetweenLines() { Line = "20", LineRule = LineSpacingRuleValues.Exact }));
                for (int pic_id = 0; pic_id < meta_data.jsonPics.Length; pic_id++)
                {

                    JsonPic pic = meta_data.jsonPics[pic_id];
                    if (pic.in_page == 1)
                    {
                        continue;
                    }
                    int index = pic.index;
                    if (Array.IndexOf(float_content_id, index) != -1)
                    {
                        AddPicWithInPara(mainPart, pp, pic);
                    }
                }
                for (int box_id = 0; box_id < meta_data.jsonTextBoxes.Length; box_id++)
                {
                    JsonTextBox tbox = meta_data.jsonTextBoxes[box_id];
                    if (tbox.in_page == 1)
                    {
                        continue;
                    }
                    int index = tbox.index;
                    if (Array.IndexOf(float_content_id, index) != -1)
                    {
                        AddTextBoxesWithInPara(pp, tbox);
                    }
                }
                for (int table_id = 0; table_id < meta_data.jsonTables.Length; table_id++)
                {
                    JsonTable table = meta_data.jsonTables[table_id];
                    if (table.in_page == 1)
                    {
                        continue;
                    }
                    int index = table.index;
                    if (Array.IndexOf(float_content_id, index) != -1)
                    {
                        Table t = AddTableWithInPara(mainPart, pp, table);
                    }
                }
                body.Append(pp);
            }
            int pad_line = para_meta_data.padLine;
            if (pad_line == -1)
            {
                Paragraph p1 = new Paragraph();
                ParagraphProperties pPr1 = new ParagraphProperties();
                float padLineSpace = (float)0.0352 * fontSize;
                //间距
                SpacingBetweenLines space1 = new SpacingBetweenLines();
                space1.Before = (Convert.ToUInt32(0)).ToString();
                space1.After = (Convert.ToUInt32(0)).ToString();
                space1.Line = (Convert.ToUInt32(20)).ToString();
                space1.LineRule = LineSpacingRuleValues.Exact;
                pPr1.Append(space1);
                p1.Append(pPr1);
                body.Append(p1);
            }
            else
            {
                for (int id = 0; id < pad_line; id++)
                {
                    Paragraph p1 = new Paragraph();
                    ParagraphProperties pPr1 = new ParagraphProperties();
                    float padLineSpace = (float)0.0352 * fontSize;
                    //间距
                    SpacingBetweenLines space1 = new SpacingBetweenLines();
                    space1.Before = (Convert.ToUInt32(0)).ToString();
                    space1.After = (Convert.ToUInt32(0)).ToString();
                    space1.Line = (Convert.ToUInt32(padLineSpace * 567)).ToString();
                    space1.LineRule = LineSpacingRuleValues.Exact;
                    pPr1.Append(space1);
                    p1.Append(pPr1);
                    body.Append(p1);
                }
            }
            if (paraAddBreak)
            {
                Paragraph p1 = new Paragraph();
                ParagraphProperties pPr1 = new ParagraphProperties();
                //间距
                SpacingBetweenLines space1 = new SpacingBetweenLines();
                space1.Before = (Convert.ToUInt32(0)).ToString();
                space1.After = (Convert.ToUInt32(0)).ToString();
                space1.Line = (Convert.ToUInt32(20)).ToString();
                space1.LineRule = LineSpacingRuleValues.Exact;
                pPr1.Append(space1);
                p1.Append(pPr1);
                Run run = new Run();
                Break break1 = new Break() { Type = BreakValues.Column };
                run.Append(break1);
                p1.Append(run);
                body.Append(p1);
            }


        }
        public static void AddParagraph(TableCell tc, JsonParagraph para_meta_data, JsonPic[] cell_pics, MainDocumentPart mainPart)
        {
            if (para_meta_data == null) {
                for (int cell_index = 0; cell_index < cell_pics.Length; cell_index++)
                {
                    JsonPic pic_data = cell_pics[cell_index];
                    float[] bbox = pic_data.bbox;
                    float position_w_from_colum = pic_data.position_w_from_colum;
                    float position_h_from_paragraph = pic_data.position_h_from_paragraph;
                    List<float> position = new List<float>() { position_w_from_colum, position_h_from_paragraph, bbox[2] - bbox[0], bbox[3] - bbox[1] };
                    string url = pic_data.url;
                    ImagePart imagePart = mainPart.AddImagePart(ImagePartType.Jpeg);
                    using (FileStream stream = new FileStream(url, FileMode.Open))
                    //using (FileStream stream = new FileStream("qq.png", FileMode.Open))
                    {
                        imagePart.FeedData(stream);
                    }
                    Drawing pic = DocUtils.CreateInlineDrawing(mainPart.GetIdOfPart(imagePart), position, shape_id);
                    Run imageRun = new Run();
                    imageRun.Append(pic);
                    tc.Append(new Paragraph(imageRun));
                    shape_id++;
                }
                return;
            }
             if (para_meta_data.cell_before_pics.Length != 0)
            {
                for (int i = 0; i < para_meta_data.cell_before_pics.Length; i++)
                {
                    int cell_index = para_meta_data.cell_before_pics[i];
                    JsonPic pic_data = cell_pics[cell_index];
                    float[] bbox = pic_data.bbox;
                    float position_w_from_colum = pic_data.position_w_from_colum;
                    float position_h_from_paragraph = pic_data.position_h_from_paragraph;
                    List<float> position = new List<float>() { position_w_from_colum, position_h_from_paragraph, bbox[2] - bbox[0], bbox[3] - bbox[1] };
                    string url = pic_data.url;
                    ImagePart imagePart = mainPart.AddImagePart(ImagePartType.Jpeg);
                    using (FileStream stream = new FileStream(url, FileMode.Open))
                    //using (FileStream stream = new FileStream("qq.png", FileMode.Open))
                    {
                        imagePart.FeedData(stream);
                    }
                    Drawing pic = DocUtils.CreateInlineDrawing(mainPart.GetIdOfPart(imagePart), position, shape_id);
                    Run imageRun = new Run();
                    imageRun.Append(pic);
                    tc.Append(new Paragraph(imageRun));
                    shape_id++;
                }
            }
            float spaceBefore = para_meta_data.spaceBefore;
            float lineSpace = para_meta_data.lineSpace;
            float firstLineLeftIndent = Math.Max(para_meta_data.firstLineLeftIndent, 0);
            float lineIndent = para_meta_data.leftIndent;

            string align = para_meta_data.align;
            Paragraph p = new Paragraph();
            ParagraphProperties pPr = new ParagraphProperties();

            //间距
            SpacingBetweenLines space = new SpacingBetweenLines();
            space.Before = (Convert.ToUInt32(spaceBefore * 567 * 0.8)).ToString();
            space.Line = (Convert.ToUInt32(lineSpace * 567)).ToString();
            space.LineRule = LineSpacingRuleValues.Exact;

            //缩进
            Indentation indentation = new Indentation();
            indentation.Left = (Convert.ToUInt32(lineIndent * 567)).ToString();
            indentation.FirstLine = (Convert.ToUInt32(firstLineLeftIndent * 567)).ToString();

            //中西文间距
            AutoSpaceDE autoSpaceDE = new AutoSpaceDE() { Val = false };
            AutoSpaceDN autoSpaceDN = new AutoSpaceDN() { Val = false };

            //对齐
            Justification justification = new Justification() { Val = JustificationValues.Left };
            if (align == "center")
            {
                justification.Val = JustificationValues.Center;
            }

            //设置段落属性
            pPr.Append(space);
            pPr.Append(indentation);
            pPr.Append(autoSpaceDE);
            pPr.Append(autoSpaceDN);
            pPr.Append(justification);
            p.Append(pPr);

            int fontSize = para_meta_data.fontSize;

            for (int id = 0; id < para_meta_data.lines.Length; id++)
            {
                JsonLine line = para_meta_data.lines[id];
                float wordRatio = line.wordRatio;
                float wordSpace = line.wordSpace;
                JsonText[] line_texts = line.texts;
                for (int text_id = 0; text_id < line_texts.Length; text_id++)
                {
                    string text = line_texts[text_id].text;
                    if (text == "")
                    {
                        continue;
                    }
                    bool isomml = text.Contains("omml");
                    if (isomml)
                    {
                        string ommlstr = text.Split("omml")[1];
                        DocUtils.AddMathFormula(p, ommlstr, fontSize);
                        // alignment.val = verticaltextalignmentvalues.baseline;
                        continue;
                    }
                    Run run = p.AppendChild(new Run());
                    RunProperties rPr = DocUtils.GetRunProperties(fontSize, para_meta_data.blackFont);
                    rPr = DocUtils.SetWordSpace(rPr, wordRatio, wordSpace);
                    if (text != "" && text[0] != '^')
                    {
                        run.AddChild(rPr);
                        Text lineText = new Text(text);
                    }
                    else if (text.Length >= 2 && text[1] == '^')
                    {
                        VerticalTextAlignment v = new VerticalTextAlignment() { Val = VerticalPositionValues.Subscript };
                        rPr.Append(v);
                        run.AddChild(rPr);
                        string new_text = "";
                        for (int j = 2; j < text.Length; j++)
                        {
                            new_text += text[j];
                        }
                        text = new_text;
                    }
                    else
                    {
                        string new_text = "";
                        for (int j = 1; j < text.Length; j++)
                        {
                            new_text += text[j];
                        }
                        VerticalTextAlignment v = new VerticalTextAlignment() { Val = VerticalPositionValues.Superscript };
                        rPr.Append(v);
                        run.AddChild(rPr);
                        text = new_text;
                    }
                    Text t = new Text(text);
                    t.Space = SpaceProcessingModeValues.Preserve;
                    run.Append(t);
                }

            }
            tc.Append(p);
            if (para_meta_data.cell_follow_pics.Length != 0)
            {
                for (int i = 0; i < para_meta_data.cell_follow_pics.Length; i++)
                {
                    int cell_index = para_meta_data.cell_follow_pics[i];
                    JsonPic pic_data = cell_pics[cell_index];
                    float[] bbox = pic_data.bbox;
                    float position_w_from_colum = pic_data.position_w_from_colum;
                    float position_h_from_paragraph = pic_data.position_h_from_paragraph;
                    List<float> position = new List<float>() { position_w_from_colum, position_h_from_paragraph, bbox[2] - bbox[0], bbox[3] - bbox[1] };
                    string url = pic_data.url;
                    ImagePart imagePart = mainPart.AddImagePart(ImagePartType.Jpeg);
                    using (FileStream stream = new FileStream(url, FileMode.Open))
                    //using (FileStream stream = new FileStream("qq.jpg", FileMode.Open))
                    {
                        imagePart.FeedData(stream);
                    }
                    //Drawing pic = DocUtils.GetAnchorPicture(mainPart.GetIdOfPart(imagePart), Path.GetFileNameWithoutExtension(url), position, shape_id, false, true, "both");
                    Drawing pic = DocUtils.CreateInlineDrawing(mainPart.GetIdOfPart(imagePart), position, shape_id);
                    Run imageRun = new Run();
                    imageRun.Append(pic);
                    tc.Append(new Paragraph(imageRun));
                    shape_id++;
                }
            }
            int pad_line = para_meta_data.padLine;
            for (int id = 0; id < pad_line; id++)
            {
                Paragraph p1 = new Paragraph();
                ParagraphProperties pPr1 = new ParagraphProperties();
                //间距
                SpacingBetweenLines space1 = new SpacingBetweenLines();
                space1.Before = (Convert.ToUInt32(0)).ToString();
                space1.After = (Convert.ToUInt32(0)).ToString();
                space1.Line = (Convert.ToUInt32(lineSpace * 567)).ToString();
                space1.LineRule = LineSpacingRuleValues.Exact;
                pPr1.Append(space1);
                p1.Append(pPr1);
                tc.Append(p1);
            }

        }

        public static void AddTextBoxes(Body body, JsonTextBox[] textboxes)
        {
            IEnumerable<Paragraph> allParagraphs = body.Elements<Paragraph>();
            int k = 0;
            Paragraph p = null;
            foreach (Paragraph tmp in allParagraphs)
            {
                if (k == floatContentParaIndex)
                {
                    p = tmp;
                    break;
                }
                k++;
            }
            if (p == null)
            {
                p = body.AppendChild(new Paragraph());
            }
            Run run = p.AppendChild(new Run());
            for (int id = 0; id < textboxes.Length; id++)
            {
                if (textboxes[id].in_page != 1)
                {
                    continue;
                }
                float[] bbox = textboxes[id].bbox;
                List<float> textboxPositin = new List<float> { bbox[0], bbox[1], bbox[2] - bbox[0] + (float)0.3, bbox[3] - bbox[1] + (float)0.3 };
                AlternateContent textbox1 = DocUtils.AddBlankTextBox(textboxPositin, shape_id, false);
                List<TextBoxContent> textboxcontents = textbox1.Descendants<TextBoxContent>().ToList();
                for (int j = 0; j < textboxcontents.Count; j++)
                {
                    int fontSize = textboxes[id].fontSize;
                    float lineSpace = fontSize * (float)1.5 * (float)0.03527;
                    Paragraph para1 = new Paragraph();
                    Justification justification = new Justification() { Val = JustificationValues.Left };
                    ParagraphProperties paragraphProperties = new ParagraphProperties();
                    SpacingBetweenLines space = new SpacingBetweenLines();
                    space.Before = (Convert.ToUInt32(0)).ToString();
                    space.Line = (Convert.ToUInt32(lineSpace * 567)).ToString();
                    space.LineRule = LineSpacingRuleValues.Exact;

                    paragraphProperties.Append(justification);
                    paragraphProperties.Append(space);
                    para1.Append(paragraphProperties);
                    Run run1 = new Run();
                    RunProperties rp = DocUtils.GetRunProperties(fontSize);
                    run1.PrependChild<RunProperties>(rp);
                    Text cur_text = new Text();
                    cur_text.Text = textboxes[id].text;
                    run1.Append(cur_text);
                    para1.Append(run1);
                    textboxcontents[j].Append(para1);
                }
                run.Append(textbox1);
                shape_id += 1;
            }
        }
        public static void AddTextBoxesWithInPara(Paragraph p, JsonTextBox textbox)
        {
            Run run = p.AppendChild(new Run());
            float[] bbox = textbox.bbox;
            float position_w_from_colum = textbox.position_w_from_colum;
            float position_h_from_paragraph = textbox.position_h_from_paragraph;
            List<float> textboxPositin = new List<float> { position_w_from_colum, position_h_from_paragraph, bbox[2] - bbox[0] + (float)0.3, bbox[3] - bbox[1] + (float)0.3 };
            AlternateContent textbox1 = DocUtils.AddBlankTextBox(textboxPositin, shape_id, false, false);
            List<TextBoxContent> textboxcontents = textbox1.Descendants<TextBoxContent>().ToList();
            for (int j = 0; j < textboxcontents.Count; j++)
            {
                int fontSize = textbox.fontSize;
                float lineSpace = fontSize * (float)1.5 * (float)0.03527;
                Paragraph para1 = new Paragraph();
                Justification justification = new Justification() { Val = JustificationValues.Left };
                ParagraphProperties paragraphProperties = new ParagraphProperties(new TextAlignment() { Val = VerticalTextAlignmentValues.Baseline });
                SpacingBetweenLines space = new SpacingBetweenLines();
                space.Before = (Convert.ToUInt32(0)).ToString();
                space.Line = (Convert.ToUInt32(lineSpace * 567)).ToString();
                space.LineRule = LineSpacingRuleValues.Exact;

                paragraphProperties.Append(justification);
                paragraphProperties.Append(space);
                para1.Append(paragraphProperties);
                Run run1 = new Run();
                RunProperties rp = DocUtils.GetRunProperties(fontSize);
                run1.PrependChild<RunProperties>(rp);
                Text cur_text = new Text();
                cur_text.Text = textbox.text;
                run1.Append(cur_text);
                para1.Append(run1);
                textboxcontents[j].Append(para1);
            }
            run.Append(textbox1);
            shape_id += 1;

        }
        public static uint imageId = 0;


        public static void AddPicWithInPara(MainDocumentPart mainPart, Paragraph p, JsonPic pic_data)
        {
            float[] bbox = pic_data.bbox;
            float position_w_from_colum = pic_data.position_w_from_colum;
            float position_h_from_paragraph = pic_data.position_h_from_paragraph;
            List<float> position = new List<float>() { position_w_from_colum, position_h_from_paragraph, bbox[2] - bbox[0], bbox[3] - bbox[1] };
            string url = pic_data.url;
            bool warp = false;
            string text_warp = "left";
            if (pic_data.warp == "warpleft")
            {
                warp = true;
            }
            else if (pic_data.warp == "warpboth") {
                warp = true;
                text_warp = "both";
            }
            ImagePart imagePart = mainPart.AddImagePart(ImagePartType.Jpeg);
            using (FileStream stream = new FileStream(url, FileMode.Open))
           // using (FileStream stream = new FileStream("qq.jpg", FileMode.Open))
            {
                imagePart.FeedData(stream);
            }
            Drawing pic = DocUtils.GetAnchorPicture(mainPart.GetIdOfPart(imagePart), Path.GetFileNameWithoutExtension(url), position, shape_id, false, warp, text_warp);
            //String rid = mainPart.GetIdOfPart(imagePart);
            //String iname = Path.GetFileNameWithoutExtension(url);
            //Drawing pic = CreateAnchorDrawing(rid, iname, (float)360000, (float)360000);

            Run run = p.AppendChild(new Run(pic));
            //run.PrependChild<RunProperties>(new RunProperties(new NoProof()));
            //run.Append(pic);
            shape_id += 1;
        }

        public static void AddFloatPictures(MainDocumentPart mainPart, Body body, JsonPic[] pictures)
        {
            IEnumerable<Paragraph> allParagraphs = body.Elements<Paragraph>();
            int k = 0;
            Paragraph p = null;
            foreach (Paragraph tmp in allParagraphs)
            {
                if (k == floatContentParaIndex)
                {
                    p = tmp;
                    break;
                }
                k += 1;
            }
            if (p == null)
            {
                p = body.AppendChild(new Paragraph());
            }
            for (int id = 0; id < pictures.Length; id++)
            {
                JsonPic pic_data = pictures[id];
                if (pictures[id].in_page != 1)
                {
                    continue;
                }
                float[] bbox = pic_data.bbox;
                List<float> position = new List<float>() { bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1] };
                string url = pic_data.url;
                ImagePart imagePart = mainPart.AddImagePart(ImagePartType.Jpeg);
                using (FileStream stream = new FileStream(url, FileMode.Open))
                {
                    imagePart.FeedData(stream);
                }
                Drawing pic = DocUtils.GetAnchorPicture(mainPart.GetIdOfPart(imagePart), Path.GetFileNameWithoutExtension(url), position, shape_id);

                Run run = p.AppendChild(new Run(pic));
                shape_id += 1;
            }
        }

        public static void AddTableInTextBox(MainDocumentPart mainPart, Body body, JsonTable[] tables)
        {
            IEnumerable<Paragraph> allParagraphs = body.Elements<Paragraph>();
            int k = 0;
            Paragraph p = null;
            foreach (Paragraph tmp in allParagraphs)
            {
                if (k == floatContentParaIndex)
                {
                    p = tmp;
                    break;
                }
                k++;
            }
            if (p == null)
            {
                p = body.AppendChild(new Paragraph());
            }
            Run run = p.AppendChild(new Run());

            for (int id = 0; id < tables.Length; id++)
            {
                if (tables[id].in_page != 1)
                {
                    continue;
                }
                float[] bbox = tables[id].bbox;
                List<float> textboxPositin = new List<float> { bbox[0], bbox[1], bbox[2] - bbox[0] + (float)0.3, bbox[3] - bbox[1] + (float)0.3 };
                AlternateContent textbox = DocUtils.AddBlankTextBox(textboxPositin, shape_id, true);
                Table table = TableUtils.AddTable(tables[id], mainPart);
                List<TextBoxContent> textBoxContents = textbox.Descendants<TextBoxContent>().ToList();
                for (int j = 0; j < textBoxContents.Count; j++)
                {
                    textBoxContents[j].Append(table.CloneNode(true));
                }
                run.Append(textbox);
                shape_id += 1;
            }
        }

        public static Table AddTableWithInPara(MainDocumentPart mainPart, Paragraph p, JsonTable table_meta_data, bool  use_txbx=true)
        {

            Run run = p.AppendChild(new Run());
            float[] bbox = table_meta_data.bbox;
            string warp_ = table_meta_data.warp;
            bool warp = false;
            float pad_w = 1;
            float pad_h = (float)0.8;
            if (warp_ == "warpboth") {
                warp = true;
                pad_w = (float)0.2;
                pad_h = (float)0.2;
            }
            float position_w_from_colum = table_meta_data.position_w_from_colum;
            float position_h_from_paragraph = table_meta_data.position_h_from_paragraph;
            //List<float> textboxPositin = new List<float> { position_w_from_colum - (float)1, position_h_from_paragraph, bbox[2] - bbox[0] + (float)2, bbox[3] - bbox[1] + (float)0.8 };
            List<float> textboxPositin = new List<float> { position_w_from_colum, position_h_from_paragraph, bbox[2] - bbox[0] + (float)pad_w, bbox[3] - bbox[1] + (float)pad_h};
            AlternateContent textbox = DocUtils.AddBlankTextBox(textboxPositin, shape_id, true, false, warp);
            Table table = TableUtils.AddTable(table_meta_data, mainPart);
            List<TextBoxContent> textBoxContents = textbox.Descendants<TextBoxContent>().ToList();
            for (int j = 0; j < textBoxContents.Count; j++)
            {
                textBoxContents[j].Append(new Paragraph(new ParagraphProperties(new SpacingBetweenLines() { Line = "20", LineRule = LineSpacingRuleValues.Exact })));
                textBoxContents[j].Append(table.CloneNode(true));
                textBoxContents[j].Append(new Paragraph(new ParagraphProperties(new SpacingBetweenLines() { Line = "20", LineRule = LineSpacingRuleValues.Exact })));
            }
            if (use_txbx)
            {
                run.Append(textbox);
                shape_id += 1;
            }
            return table;

        }
    }
}


