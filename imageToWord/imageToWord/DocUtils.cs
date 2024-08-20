using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;


using A = DocumentFormat.OpenXml.Drawing;
using A14 = DocumentFormat.OpenXml.Office2010.Drawing;
using Ovml = DocumentFormat.OpenXml.Vml.Office;
using V = DocumentFormat.OpenXml.Vml;
using Wp = DocumentFormat.OpenXml.Drawing.Wordprocessing;
using Wps = DocumentFormat.OpenXml.Office2010.Word.DrawingShape;
using PIC = DocumentFormat.OpenXml.Drawing.Pictures;
using Math = DocumentFormat.OpenXml.Math;
namespace imageToWord
{
    class DocUtils
    {
        public static Document GetInitDocument() {
            Document doc = new Document();
            doc.AddNamespaceDeclaration("wpc", "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas");
            doc.AddNamespaceDeclaration("mc", "http://schemas.openxmlformats.org/markup-compatibility/2006");
            doc.AddNamespaceDeclaration("o", "urn:schemas-microsoft-com:office:office");
            doc.AddNamespaceDeclaration("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships");
            doc.AddNamespaceDeclaration("m", "http://schemas.openxmlformats.org/officeDocument/2006/math");
            doc.AddNamespaceDeclaration("v", "urn:schemas-microsoft-com:vml");
            doc.AddNamespaceDeclaration("wp14", "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing");
            doc.AddNamespaceDeclaration("wp", "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing");
            doc.AddNamespaceDeclaration("w10", "urn:schemas-microsoft-com:office:word");
            doc.AddNamespaceDeclaration("w", "http://schemas.openxmlformats.org/wordprocessingml/2006/main");
            doc.AddNamespaceDeclaration("w14", "http://schemas.microsoft.com/office/word/2010/wordml");
            doc.AddNamespaceDeclaration("w15", "http://schemas.microsoft.com/office/word/2012/wordml");
            doc.AddNamespaceDeclaration("wpg", "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup");
            doc.AddNamespaceDeclaration("wpi", "http://schemas.microsoft.com/office/word/2010/wordprocessingInk");
            doc.AddNamespaceDeclaration("wne", "http://schemas.microsoft.com/office/word/2006/wordml");
            doc.AddNamespaceDeclaration("wps", "http://schemas.microsoft.com/office/word/2010/wordprocessingShape");
            doc.AddNamespaceDeclaration("pic", "http://schemas.openxmlformats.org/drawingml/2006/picture");
            doc.AddNamespaceDeclaration("a14", "http://schemas.microsoft.com/office/drawing/2010/main");
            return doc;
        }

        public static void AddSettings(MainDocumentPart mainPart) {
            DocumentSettingsPart settingsPart = mainPart.AddNewPart<DocumentSettingsPart>();
            settingsPart.Settings = new Settings(
               new Compatibility(
                   new CompatibilitySetting()
                   {
                       Name = new EnumValue<CompatSettingNameValues>(CompatSettingNameValues.CompatibilityMode),
                       Val = new StringValue("14"),
                       Uri = new StringValue("http://schemas.microsoft.com/office/word")
                   }
               )
               { BalanceSingleByteDoubleByteWidth= new BalanceSingleByteDoubleByteWidth() { Val=false} }
            );
            settingsPart.Settings.Save();
        }

        public static void AddSettingsFromSettingXml(MainDocumentPart mainPart)
        {
            DocumentSettingsPart settingsPart = mainPart.AddNewPart<DocumentSettingsPart>();
            FileStream settingsTemplate = new FileStream("settings.xml", FileMode.Open, FileAccess.Read);
            settingsPart.FeedData(settingsTemplate);
            settingsPart.Settings.Save();
        }

        //public static void AddWebSettingsFromSettingXml(MainDocumentPart mainPart)
        //{
        //    WebSettingsPart settingsPart = mainPart.AddNewPart<WebSettingsPart>();
        //    FileStream settingsTemplate = new FileStream("webSettings.xml", FileMode.Open, FileAccess.Read);
        //    settingsPart.FeedData(settingsTemplate);
        //    WebSettings webSettings = new WebSettings();
        //    webSettings.WebSettingsPart = settingsPart;
        //    settingsPart.Save();
        //}

        public static void MoveSectionPr(Body body) {
            IEnumerable<SectionProperties> SectionPropertys = body.Descendants<SectionProperties>();
            //若当前的全局sectPr存在，则转移为最后一个段落的局部sectPr
            if (SectionPropertys.Count() != 0)
            {
                SectionProperties lastSectionProperty = SectionPropertys.Last();
                body.RemoveChild(lastSectionProperty);
                IEnumerable<Paragraph> paras = body.Elements<Paragraph>();
                Paragraph last_p = null;
                if (paras.Count() == 0)
                {
                    last_p = new Paragraph();
                }
                else
                {
                    last_p = body.Elements<Paragraph>().Last();
                }

                ParagraphProperties pPr = last_p.ParagraphProperties;
                if (pPr == null)
                {
                    pPr = new ParagraphProperties();
                    pPr.Append(lastSectionProperty);
                    last_p.Append(pPr);
                }
                else
                {
                    pPr.Append(lastSectionProperty);
                }
            }
        }

        public static void AddStylesFromXml(WordprocessingDocument doc)
        {
            StyleDefinitionsPart stylePart;
            stylePart = doc.MainDocumentPart.AddNewPart<StyleDefinitionsPart>();
            FileStream stylessTemplate = new FileStream("styles.xml", FileMode.Open, FileAccess.Read);
            stylePart.FeedData(stylessTemplate);
            stylePart.Styles.Save();
        }

        public static StyleDefinitionsPart AddStylesPartToPackage(WordprocessingDocument doc)
        {
            StyleDefinitionsPart part;
            part = doc.MainDocumentPart.AddNewPart<StyleDefinitionsPart>();
            Styles root = new Styles();
            root.Save(part);
            
            return part;
        }

        public static void CreateAndAddCharacterStyle(StyleDefinitionsPart styleDefinitionsPart)
        {

            // Get access to the root element of the styles part.
            Styles styles = styleDefinitionsPart.Styles;

            // Create a new character style and specify some of the attributes.
            Style style = new Style()
            {
                Type = StyleValues.Paragraph,
                //CustomStyle = true
            };

            // Create the StyleRunProperties object and specify some of the run properties.
            StyleRunProperties styleRunProperties1 = new StyleRunProperties() {};
          
            RunFonts font1 = new RunFonts()
            {
                Ascii = "TimeNewRomans",
                EastAsia = "宋体",
            };
            FontSize sz = new FontSize() { Val = "16" };
            styleRunProperties1.Append(font1);
            styleRunProperties1.Append(sz);
            // Add the run properties to the style.
            style.Append(styleRunProperties1);
            // Add the style to the styles part.
            styles.Append(style);
        }

        public static RunProperties GetRunProperties(int fontSize, int blackFont=0)
        {
            RunProperties runProperties = new RunProperties();
            RunFonts font = new RunFonts();
            font.EastAsia = "宋体";
            font.Ascii = "Times New Roman";
            font.HighAnsi = "宋体";
            FontSize size = new FontSize();
            size.Val = new StringValue((fontSize * 2).ToString());
            runProperties.Append(size);
            runProperties.Append(font);
            if (blackFont == 1) {
                runProperties.Append(new Bold());
                runProperties.Append(new BoldComplexScript());
            }
            //Kern kern = new Kern() { Val=0};
            //runProperties.Append(kern);

            return runProperties;
        }

        public static RunProperties SetWordSpace(RunProperties runProperties, float wordRatio, float wordSpace)
        {
            Spacing spacing = new Spacing() { Val = Convert.ToInt32(20 * wordSpace) };
            //Spacing spacing = new Spacing() { Val = Convert.ToInt32(567* (wordSpace)) };
            runProperties.Append(spacing);

            //CharacterScale characterScale = new CharacterScale() { Val = Convert.ToInt32(wordRatio) };
            CharacterScale characterScale = new CharacterScale() { Val = Convert.ToInt32(100) };
            runProperties.Append(characterScale);
            return runProperties;
        }

        public static AlternateContent AddBlankTextBox(List<float> position, int box_id, bool isTable, bool coordFromPage=true, bool warp=false)
        {
            //传入文本框的坐标，以cm为单位，返回的为空白无边框无填充box
            //scale为cm和xml中坐标度量的尺度
            int scale = 360000;
            Int32 X1 = Convert.ToInt32(position[0] * scale);
            Int32 Y1 = Convert.ToInt32(position[1] * scale);
            UInt32 Width = Convert.ToUInt32(position[2] * scale);
            UInt32 Height = Convert.ToUInt32(position[3] * scale);

            AlternateContent alternateContent1 = new AlternateContent();
            AlternateContentChoice alternateContentChoice1 = new AlternateContentChoice() { Requires = "wps" };
            Drawing drawing1 = new Drawing();
            Wp.Anchor anchor1 = new Wp.Anchor() { DistanceFromTop = (UInt32Value)0U, DistanceFromBottom = (UInt32Value)0U, DistanceFromLeft = (UInt32Value)114300U, DistanceFromRight = (UInt32Value)114300U, SimplePos = false, RelativeHeight = (UInt32Value)251659264U, BehindDoc = false, Locked = false, LayoutInCell = true, AllowOverlap = true };
            Wp.SimplePosition simplePosition1 = new Wp.SimplePosition() { X = 0L, Y = 0L };

            Wp.HorizontalPosition horizontalPosition1 = new Wp.HorizontalPosition() { RelativeFrom = Wp.HorizontalRelativePositionValues.Page };
            if (!coordFromPage) {
                horizontalPosition1.RelativeFrom = Wp.HorizontalRelativePositionValues.Column;
            }
            Wp.PositionOffset positionOffset1 = new Wp.PositionOffset();
            positionOffset1.Text = X1.ToString();

            horizontalPosition1.Append(positionOffset1);

            Wp.VerticalPosition verticalPosition1 = new Wp.VerticalPosition() { RelativeFrom = Wp.VerticalRelativePositionValues.Page };
            if (!coordFromPage)
            {
                verticalPosition1.RelativeFrom = Wp.VerticalRelativePositionValues.Paragraph;
            }
            Wp.PositionOffset positionOffset2 = new Wp.PositionOffset();
            positionOffset2.Text = Y1.ToString();

            verticalPosition1.Append(positionOffset2);
            Wp.Extent extent1 = new Wp.Extent() { Cx = Width, Cy = Height };
            Wp.EffectExtent effectExtent1 = new Wp.EffectExtent() { LeftEdge = 0L, TopEdge = 0L, RightEdge = 0L, BottomEdge = 0L };
            Wp.WrapNone wrapNone1 = new Wp.WrapNone();
            Wp.WrapSquare wrapSquare = new Wp.WrapSquare()
            {
                WrapText = Wp.WrapTextValues.BothSides
            };
            Wp.DocProperties docProperties1 = new Wp.DocProperties() { Id = Convert.ToUInt32(box_id), Name = "textbox" + box_id.ToString() };
            Wp.NonVisualGraphicFrameDrawingProperties nonVisualGraphicFrameDrawingProperties1 = new Wp.NonVisualGraphicFrameDrawingProperties();

            A.Graphic graphic1 = new A.Graphic();
            graphic1.AddNamespaceDeclaration("a", "http://schemas.openxmlformats.org/drawingml/2006/main");

            A.GraphicData graphicData1 = new A.GraphicData() { Uri = "http://schemas.microsoft.com/office/word/2010/wordprocessingShape" };

            Wps.WordprocessingShape wordprocessingShape1 = new Wps.WordprocessingShape();

            Wps.NonVisualDrawingShapeProperties nonVisualDrawingShapeProperties1 = new Wps.NonVisualDrawingShapeProperties() { TextBox = true };
            wordprocessingShape1.Append(nonVisualDrawingShapeProperties1);

            Wps.ShapeProperties shapeProperties1 = new Wps.ShapeProperties();

            A.Transform2D transform2D1 = new A.Transform2D();
            A.Offset offset1 = new A.Offset() { X = 0L, Y =0L };
            A.Extents extents1 = new A.Extents() { Cx = Width, Cy = Height };

            transform2D1.Append(offset1);
            transform2D1.Append(extents1);

            A.PresetGeometry presetGeometry1 = new A.PresetGeometry() { Preset = A.ShapeTypeValues.Rectangle };
            A.AdjustValueList adjustValueList1 = new A.AdjustValueList();

            presetGeometry1.Append(adjustValueList1);
            A.NoFill noFill1 = new A.NoFill();

            A.Outline outline1 = new A.Outline() { Width = 6350 };
            A.NoFill noFill2 = new A.NoFill();

            outline1.Append(noFill2);
            A.EffectList effectList1 = new A.EffectList();

            A.ShapePropertiesExtensionList shapePropertiesExtensionList1 = new A.ShapePropertiesExtensionList();

            A.ShapePropertiesExtension shapePropertiesExtension1 = new A.ShapePropertiesExtension() { Uri = "{91240B29-F687-4F45-9708-019B960494DF}" };

            A14.HiddenLineProperties hiddenLineProperties1 = new A14.HiddenLineProperties() { Width = 6350 };
            hiddenLineProperties1.AddNamespaceDeclaration("a14", "http://schemas.microsoft.com/office/drawing/2010/main");

            A.SolidFill solidFill1 = new A.SolidFill();
            A.PresetColor presetColor1 = new A.PresetColor() { Val = A.PresetColorValues.Black };

            solidFill1.Append(presetColor1);

            hiddenLineProperties1.Append(solidFill1);

            shapePropertiesExtension1.Append(hiddenLineProperties1);

            shapePropertiesExtensionList1.Append(shapePropertiesExtension1);

            shapeProperties1.Append(transform2D1);
            shapeProperties1.Append(presetGeometry1);
            shapeProperties1.Append(noFill1);
            shapeProperties1.Append(outline1);
            wordprocessingShape1.Append(shapeProperties1);


            Wps.ShapeStyle shapeStyle1 = new Wps.ShapeStyle();

            A.LineReference lineReference1 = new A.LineReference() { Index = (UInt32Value)0U };
            A.SchemeColor schemeColor1 = new A.SchemeColor() { Val = A.SchemeColorValues.Accent1 };

            lineReference1.Append(schemeColor1);

            A.FillReference fillReference1 = new A.FillReference() { Index = (UInt32Value)0U };
            A.SchemeColor schemeColor2 = new A.SchemeColor() { Val = A.SchemeColorValues.Accent1 };

            fillReference1.Append(schemeColor2);

            A.EffectReference effectReference1 = new A.EffectReference() { Index = (UInt32Value)0U };
            A.SchemeColor schemeColor3 = new A.SchemeColor() { Val = A.SchemeColorValues.Accent1 };

            effectReference1.Append(schemeColor3);

            A.FontReference fontReference1 = new A.FontReference() { Index = A.FontCollectionIndexValues.Minor };
            A.SchemeColor schemeColor4 = new A.SchemeColor() { Val = A.SchemeColorValues.Dark1 };

            fontReference1.Append(schemeColor4);

            shapeStyle1.Append(lineReference1);
            shapeStyle1.Append(fillReference1);
            shapeStyle1.Append(effectReference1);
            shapeStyle1.Append(fontReference1);

            Wps.TextBoxInfo2 textBoxInfo21 = new Wps.TextBoxInfo2();

            TextBoxContent textBoxContent1 = new TextBoxContent();

            Paragraph paragraph2 = new Paragraph() { RsidParagraphMarkRevision = "00FA5335", RsidParagraphAddition = "00FA5335", RsidParagraphProperties = "00FA5335", RsidRunAdditionDefault = "00FA5335" };

            ParagraphProperties paragraphProperties2 = new ParagraphProperties();
            SpacingBetweenLines spacingBetweenLines2 = new SpacingBetweenLines() { After = "0", Line = "240", LineRule = LineSpacingRuleValues.Auto };
            Justification justification2 = new Justification() { Val = JustificationValues.Left };

            ParagraphMarkRunProperties paragraphMarkRunProperties2 = new ParagraphMarkRunProperties();
            FontSize fontSize3 = new FontSize() { Val = "20" };

            paragraphMarkRunProperties2.Append(fontSize3);

            paragraphProperties2.Append(spacingBetweenLines2);
            paragraphProperties2.Append(justification2);
            paragraphProperties2.Append(paragraphMarkRunProperties2);
            paragraph2.Append(paragraphProperties2);
            Run run2 = new Run() { RsidRunProperties = "00FA5335" };
            paragraph2.Append(run2);
            textBoxInfo21.Append(textBoxContent1);

            Wps.TextBodyProperties textBodyProperties1 = new Wps.TextBodyProperties() { Rotation = 0, UseParagraphSpacing = false, Vertical = A.TextVerticalValues.Horizontal, Wrap = A.TextWrappingValues.Square, LeftInset = 0, TopInset = 0, RightInset = 0, BottomInset = 0, ColumnCount = 1, ColumnSpacing = 0, RightToLeftColumns = false, FromWordArt = false, Anchor = A.TextAnchoringTypeValues.Top, AnchorCenter = false, ForceAntiAlias = false, CompatibleLineSpacing = true };

            A.PresetTextWrap presetTextWrap1 = new A.PresetTextWrap() { Preset = A.TextShapeValues.TextNoShape };
            A.AdjustValueList adjustValueList2 = new A.AdjustValueList();

            presetTextWrap1.Append(adjustValueList2);
            if (isTable)
            {
                A.NoAutoFit noAutoFit1 = new A.NoAutoFit();
                textBodyProperties1.Append(noAutoFit1);
            }
            else
            {
                //A.NoAutoFit noAutoFit1 = new A.NoAutoFit();
                A.ShapeAutoFit shapeAutoFit = new A.ShapeAutoFit();
                textBodyProperties1.Append(shapeAutoFit);
            }


            //textBodyProperties1.Append(presetTextWrap1);

            wordprocessingShape1.Append(textBoxInfo21);
            wordprocessingShape1.Append(textBodyProperties1);

            graphicData1.Append(wordprocessingShape1);

            graphic1.Append(graphicData1);

            anchor1.Append(simplePosition1);
            anchor1.Append(horizontalPosition1);
            anchor1.Append(verticalPosition1);
            anchor1.Append(extent1);
            anchor1.Append(effectExtent1);
            if (warp)
            {
                anchor1.Append(wrapSquare);
            }
            else {
                anchor1.Append(wrapNone1);
            }
            anchor1.Append(docProperties1);
            anchor1.Append(nonVisualGraphicFrameDrawingProperties1);
            anchor1.Append(graphic1);

            drawing1.Append(anchor1);

            alternateContentChoice1.Append(drawing1);

            AlternateContentFallback alternateContentFallback1 = new AlternateContentFallback();

            Picture picture1 = new Picture();

            V.Shapetype shapetype1 = new V.Shapetype() { Id = "_x0000_t202", CoordinateSize = "21600,21600", OptionalNumber = 202, EdgePath = "m,l,21600r21600,l21600,xe" };
            V.Stroke stroke1 = new V.Stroke() { JoinStyle = V.StrokeJoinStyleValues.Miter };
            V.Path path1 = new V.Path() { AllowGradientShape = true, ConnectionPointType = Ovml.ConnectValues.Rectangle };

            shapetype1.Append(stroke1);
            shapetype1.Append(path1);

            //V.Shape shape1 = new V.Shape() { Id = "textbox 1", Style = "position:absolute;left:0;text-align:left;margin-left:82.25pt;margin-top:50.35pt;width:174.5;height:96.75;z-index:251659264;visibility:visible;mso-wrap-style:square;mso-wrap-distance-left:9pt;mso-wrap-distance-top:0;mso-wrap-distance-right:9pt;mso-wrap-distance-bottom:0;mso-position-horizontal:absolute;mso-position-horizontal-relative:text;mso-position-vertical:absolute;mso-position-vertical-relative:text;v-text-anchor:top", OptionalString = "_x0000_s1026", Filled = false, Stroked = false, StrokeWeight = ".5pt", Type = "#_x0000_t202", EncodedPackage = "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\nbWxQSwUGAAAAAAQABADzAAAAaAYAAAAA\n" };
            //V.Shape shape1 = new V.Shape() { Id = "textbox" + box_id, Style =  string.Format("position:absolute;left:0pt;margin-left:{0:F1}cm;margin-top:{1:F1}cm;height:{3:F1}cm;width:{2:F1}cm;mso-position-horizontal-relative:page;mso-position-vertical-relative:page;", position[0], position[1], position[2], position[3]), OptionalString = "_x0000_s1026", Filled = false, Stroked = false, StrokeWeight = ".5pt", Type = "#_x0000_t202", EncodedPackage = "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\nbWxQSwUGAAAAAAQABADzAAAAaAYAAAAA\n" };
            V.Shape shape1 = new V.Shape() { Id = "textbox" + box_id, Style = string.Format("position:absolute;left:0pt;margin-left:{0:F1}cm;margin-top:{1:F1}cm;height:{3:F1}cm;width:{2:F1}cm;mso-position-horizontal-relative:colum;mso-position-vertical-relative:paragraph;", position[0], position[1], position[2], position[3]), OptionalString = "_x0000_s1026", Filled = false, Stroked = false, StrokeWeight = ".5pt", Type = "#_x0000_t202", EncodedPackage = "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 LnhtbFBLBQYAAAAABAAEAPMAAAC/BQAAAAA= " };
            V.Fill fill1 = new V.Fill() { DetectMouseClick = true };
            V.TextBox textBox1 = new V.TextBox();
            if (isTable == false)
            {
                textBox1 = new V.TextBox() { Style = "mso-fit-shape-to-text:t" };
            }

            TextBoxContent textBoxContent2 = new TextBoxContent();

            Paragraph paragraph5 = new Paragraph() { RsidParagraphMarkRevision = "00FA5335", RsidParagraphAddition = "00FA5335", RsidParagraphProperties = "00FA5335", RsidRunAdditionDefault = "00FA5335" };

            ParagraphProperties paragraphProperties5 = new ParagraphProperties();
            SpacingBetweenLines spacingBetweenLines5 = new SpacingBetweenLines() { After = "0", Line = "240", LineRule = LineSpacingRuleValues.Exact };
            Justification justification5 = new Justification() { Val = JustificationValues.Left };

            ParagraphMarkRunProperties paragraphMarkRunProperties5 = new ParagraphMarkRunProperties();

            paragraphProperties5.Append(spacingBetweenLines5);
            paragraphProperties5.Append(justification5);
            paragraphProperties5.Append(paragraphMarkRunProperties5);

            paragraph5.Append(paragraphProperties5);
            Run run5 = new Run() { RsidRunProperties = "00FA5335" };
            paragraph5.Append(run5);
            textBox1.Append(textBoxContent2);
            shape1.Append(fill1);
            shape1.Append(textBox1);
            picture1.Append(shapetype1);
            picture1.Append(shape1);
            alternateContentFallback1.Append(picture1);
            alternateContent1.Append(alternateContentChoice1);
            alternateContent1.Append(alternateContentFallback1);
            return alternateContent1;
        }

        public static Drawing AddSimpleBlankTextBox(List<float> position, int box_id, bool isTable, bool coordFromPage = true)
        {
            //传入文本框的坐标，以cm为单位，返回的为空白无边框无填充box
            //scale为cm和xml中坐标度量的尺度
            int scale = 360000;
            Int32 X1 = Convert.ToInt32(position[0] * scale);
            Int32 Y1 = Convert.ToInt32(position[1] * scale);
            UInt32 Width = Convert.ToUInt32(position[2] * scale);
            UInt32 Height = Convert.ToUInt32(position[3] * scale);

            Drawing drawing1 = new Drawing();
            Wp.Anchor anchor1 = new Wp.Anchor() { DistanceFromTop = (UInt32Value)0U, DistanceFromBottom = (UInt32Value)0U, DistanceFromLeft = (UInt32Value)114300U, DistanceFromRight = (UInt32Value)114300U, SimplePos = false, RelativeHeight = (UInt32Value)251659264U, BehindDoc = false, Locked = true, LayoutInCell = true, AllowOverlap = true };
            Wp.SimplePosition simplePosition1 = new Wp.SimplePosition() { X = 0L, Y = 0L };

            Wp.HorizontalPosition horizontalPosition1 = new Wp.HorizontalPosition() { RelativeFrom = Wp.HorizontalRelativePositionValues.Page };
            if (!coordFromPage)
            {
                horizontalPosition1.RelativeFrom = Wp.HorizontalRelativePositionValues.Column;
            }
            Wp.PositionOffset positionOffset1 = new Wp.PositionOffset();
            positionOffset1.Text = X1.ToString();

            horizontalPosition1.Append(positionOffset1);

            Wp.VerticalPosition verticalPosition1 = new Wp.VerticalPosition() { RelativeFrom = Wp.VerticalRelativePositionValues.Page };
            if (!coordFromPage)
            {
                verticalPosition1.RelativeFrom = Wp.VerticalRelativePositionValues.Paragraph;
            }
            Wp.PositionOffset positionOffset2 = new Wp.PositionOffset();
            positionOffset2.Text = Y1.ToString();

            verticalPosition1.Append(positionOffset2);
            Wp.Extent extent1 = new Wp.Extent() { Cx = Width, Cy = Height };
            Wp.EffectExtent effectExtent1 = new Wp.EffectExtent() { LeftEdge = 0L, TopEdge = 0L, RightEdge = 0L, BottomEdge = 0L };
            Wp.WrapNone wrapNone1 = new Wp.WrapNone();
            Wp.DocProperties docProperties1 = new Wp.DocProperties() { Id = Convert.ToUInt32(box_id), Name = "textbox" + box_id.ToString() };
            Wp.NonVisualGraphicFrameDrawingProperties nonVisualGraphicFrameDrawingProperties1 = new Wp.NonVisualGraphicFrameDrawingProperties();

            A.Graphic graphic1 = new A.Graphic();
            graphic1.AddNamespaceDeclaration("a", "http://schemas.openxmlformats.org/drawingml/2006/main");

            A.GraphicData graphicData1 = new A.GraphicData() { Uri = "http://schemas.microsoft.com/office/word/2010/wordprocessingShape" };

            Wps.WordprocessingShape wordprocessingShape1 = new Wps.WordprocessingShape();

            Wps.NonVisualDrawingShapeProperties nonVisualDrawingShapeProperties1 = new Wps.NonVisualDrawingShapeProperties() { TextBox = true };
            wordprocessingShape1.Append(nonVisualDrawingShapeProperties1);

            Wps.ShapeProperties shapeProperties1 = new Wps.ShapeProperties();

            A.Transform2D transform2D1 = new A.Transform2D();
            A.Offset offset1 = new A.Offset() { X = 0L, Y = 0L };
            A.Extents extents1 = new A.Extents() { Cx = Width, Cy = Height };

            transform2D1.Append(offset1);
            transform2D1.Append(extents1);

            A.PresetGeometry presetGeometry1 = new A.PresetGeometry() { Preset = A.ShapeTypeValues.Rectangle };
            A.AdjustValueList adjustValueList1 = new A.AdjustValueList();

            presetGeometry1.Append(adjustValueList1);
            A.NoFill noFill1 = new A.NoFill();

            A.Outline outline1 = new A.Outline() { Width = 6350 };
            A.NoFill noFill2 = new A.NoFill();

            outline1.Append(noFill2);
            A.EffectList effectList1 = new A.EffectList();

            A.ShapePropertiesExtensionList shapePropertiesExtensionList1 = new A.ShapePropertiesExtensionList();

            A.ShapePropertiesExtension shapePropertiesExtension1 = new A.ShapePropertiesExtension() { Uri = "{91240B29-F687-4F45-9708-019B960494DF}" };

            A14.HiddenLineProperties hiddenLineProperties1 = new A14.HiddenLineProperties() { Width = 6350 };
            hiddenLineProperties1.AddNamespaceDeclaration("a14", "http://schemas.microsoft.com/office/drawing/2010/main");

            A.SolidFill solidFill1 = new A.SolidFill();
            A.PresetColor presetColor1 = new A.PresetColor() { Val = A.PresetColorValues.Black };

            solidFill1.Append(presetColor1);

            hiddenLineProperties1.Append(solidFill1);

            shapePropertiesExtension1.Append(hiddenLineProperties1);

            shapePropertiesExtensionList1.Append(shapePropertiesExtension1);

            shapeProperties1.Append(transform2D1);
            shapeProperties1.Append(presetGeometry1);
            shapeProperties1.Append(noFill1);
            shapeProperties1.Append(outline1);
            wordprocessingShape1.Append(shapeProperties1);


            Wps.ShapeStyle shapeStyle1 = new Wps.ShapeStyle();

            A.LineReference lineReference1 = new A.LineReference() { Index = (UInt32Value)0U };
            A.SchemeColor schemeColor1 = new A.SchemeColor() { Val = A.SchemeColorValues.Accent1 };

            lineReference1.Append(schemeColor1);

            A.FillReference fillReference1 = new A.FillReference() { Index = (UInt32Value)0U };
            A.SchemeColor schemeColor2 = new A.SchemeColor() { Val = A.SchemeColorValues.Accent1 };

            fillReference1.Append(schemeColor2);

            A.EffectReference effectReference1 = new A.EffectReference() { Index = (UInt32Value)0U };
            A.SchemeColor schemeColor3 = new A.SchemeColor() { Val = A.SchemeColorValues.Accent1 };

            effectReference1.Append(schemeColor3);

            A.FontReference fontReference1 = new A.FontReference() { Index = A.FontCollectionIndexValues.Minor };
            A.SchemeColor schemeColor4 = new A.SchemeColor() { Val = A.SchemeColorValues.Dark1 };

            fontReference1.Append(schemeColor4);

            shapeStyle1.Append(lineReference1);
            shapeStyle1.Append(fillReference1);
            shapeStyle1.Append(effectReference1);
            shapeStyle1.Append(fontReference1);

            Wps.TextBoxInfo2 textBoxInfo21 = new Wps.TextBoxInfo2();

            TextBoxContent textBoxContent1 = new TextBoxContent();

            Paragraph paragraph2 = new Paragraph() { RsidParagraphMarkRevision = "00FA5335", RsidParagraphAddition = "00FA5335", RsidParagraphProperties = "00FA5335", RsidRunAdditionDefault = "00FA5335" };

            ParagraphProperties paragraphProperties2 = new ParagraphProperties();
            SpacingBetweenLines spacingBetweenLines2 = new SpacingBetweenLines() { After = "0", Line = "240", LineRule = LineSpacingRuleValues.Auto };
            Justification justification2 = new Justification() { Val = JustificationValues.Left };

            ParagraphMarkRunProperties paragraphMarkRunProperties2 = new ParagraphMarkRunProperties();
            FontSize fontSize3 = new FontSize() { Val = "20" };

            paragraphMarkRunProperties2.Append(fontSize3);

            paragraphProperties2.Append(spacingBetweenLines2);
            paragraphProperties2.Append(justification2);
            paragraphProperties2.Append(paragraphMarkRunProperties2);
            paragraph2.Append(paragraphProperties2);
            Run run2 = new Run() { RsidRunProperties = "00FA5335" };
            paragraph2.Append(run2);
            textBoxInfo21.Append(textBoxContent1);

            Wps.TextBodyProperties textBodyProperties1 = new Wps.TextBodyProperties() { Rotation = 0, UseParagraphSpacing = false, 
                Vertical = A.TextVerticalValues.Horizontal, Wrap = A.TextWrappingValues.Square, LeftInset = 0, TopInset = 0, RightInset = 0, 
                BottomInset = 0, ColumnCount = 1, ColumnSpacing = 0, RightToLeftColumns = false, FromWordArt = false, Anchor = A.TextAnchoringTypeValues.Top, 
                AnchorCenter = false, ForceAntiAlias = false, CompatibleLineSpacing = true, VerticalOverflow=A.TextVerticalOverflowValues.Overflow,
                HorizontalOverflow = A.TextHorizontalOverflowValues.Overflow
            };

            A.PresetTextWrap presetTextWrap1 = new A.PresetTextWrap() { Preset = A.TextShapeValues.TextNoShape };
            A.AdjustValueList adjustValueList2 = new A.AdjustValueList();

            presetTextWrap1.Append(adjustValueList2);
            if (isTable)
            {
                A.NoAutoFit noAutoFit1 = new A.NoAutoFit();
                textBodyProperties1.Append(noAutoFit1);
            }
            else
            {
                A.ShapeAutoFit shapeAutoFit = new A.ShapeAutoFit();
                textBodyProperties1.Append(shapeAutoFit);
            }
            wordprocessingShape1.Append(textBoxInfo21);
            wordprocessingShape1.Append(textBodyProperties1);

            graphicData1.Append(wordprocessingShape1);

            graphic1.Append(graphicData1);

            anchor1.Append(simplePosition1);
            anchor1.Append(horizontalPosition1);
            anchor1.Append(verticalPosition1);
            anchor1.Append(extent1);
            anchor1.Append(effectExtent1);
            anchor1.Append(wrapNone1);
            anchor1.Append(docProperties1);
            anchor1.Append(nonVisualGraphicFrameDrawingProperties1);
            anchor1.Append(graphic1);
            drawing1.Append(anchor1);
            return drawing1;
        }
        //浮动式图片
        public static Drawing GetAnchorPicture(String imagePartId, String imageName, List<float> position, int id, bool coordFromPage=true, bool warp=false, string text_warp="none")
        {
            int scale = 360000;
            Int32 X1 = Convert.ToInt32(position[0] * scale);
            Int32 Y1 = Convert.ToInt32(position[1] * scale);
            UInt32 Width = Convert.ToUInt32(position[2] * scale);
            UInt32 Height = Convert.ToUInt32(position[3] * scale);
            Drawing _drawing = new Drawing();
            Wp.Anchor _anchor = new Wp.Anchor()
            {
                DistanceFromTop = (UInt32Value)0U,
                DistanceFromBottom = (UInt32Value)0U,
                DistanceFromLeft = (UInt32Value)0U,
                DistanceFromRight = (UInt32Value)0U,
                SimplePos = false,
                RelativeHeight = (UInt32Value)0U,
                BehindDoc = false,
                Locked = false,
                LayoutInCell = true,
                AllowOverlap = true,
               //EditId = "44CEF5E4",
               // AnchorId = "44803ED1"
            };
            Wp.SimplePosition _spos = new Wp.SimplePosition()
            {
                X = 0L,
                Y = 0L
            };

            Wp.HorizontalPosition _hp = new Wp.HorizontalPosition()
            {
                RelativeFrom = Wp.HorizontalRelativePositionValues.Page
            };
            
            Wp.PositionOffset _hPO = new Wp.PositionOffset();
            _hPO.Text = X1.ToString();
            _hp.Append(_hPO);

            Wp.VerticalPosition _vp = new Wp.VerticalPosition()
            {
                RelativeFrom = Wp.VerticalRelativePositionValues.Page
            };
            if (!coordFromPage) {
                _hp.RelativeFrom = Wp.HorizontalRelativePositionValues.Column;
                _vp.RelativeFrom = Wp.VerticalRelativePositionValues.Paragraph;
            }
            //if (warp) {
            //    _hp.RelativeFrom = Wp.HorizontalRelativePositionValues.Margin;
            //    _hp.HorizontalAlignment = new Wp.HorizontalAlignment("right");
            //}
            Wp.PositionOffset _vPO = new Wp.PositionOffset();
            _vPO.Text = Y1.ToString();
            _vp.Append(_vPO);

            Wp.Extent _e = new Wp.Extent()
            {
                Cx = Width,
                Cy = Height
            };

            Wp.EffectExtent _ee = new Wp.EffectExtent()
            {
                LeftEdge = 0L,
                TopEdge = 0L,
                RightEdge = 0L,
                BottomEdge = 0L
            };
            Wp.WrapNone wrapNone = new Wp.WrapNone();
            Wp.WrapSquare _wp = new Wp.WrapSquare()
            {
                 WrapText = Wp.WrapTextValues.Left
            };
            if (text_warp == "both")
            {
                _wp.WrapText = Wp.WrapTextValues.BothSides;
            }
            else if (text_warp == "right")
            {
                _wp.WrapText = Wp.WrapTextValues.Right;
            }
            //else {
            //    _wp.WrapText = Wp.WrapTextValues.BothSides;
            //}
                Wp.WrapPolygon _wpp = new Wp.WrapPolygon()
            {
                Edited = false
            };
            Wp.StartPoint _sp = new Wp.StartPoint()
            {
                X = 0L,
                Y = 0L
            };
            Wp.NonVisualGraphicFrameDrawingProperties nv = new Wp.NonVisualGraphicFrameDrawingProperties();
            Wp.LineTo _l1 = new Wp.LineTo() { X = 0L, Y = 0L }; 
            Wp.LineTo _l2 = new Wp.LineTo() { X = 0L, Y = 0L };
            Wp.LineTo _l3 = new Wp.LineTo() { X = 0L, Y = 0L };
            Wp.LineTo _l4 = new Wp.LineTo() { X = 0L, Y = 0L };

            _wpp.Append(_sp);
            _wpp.Append(_l1);
            _wpp.Append(_l2);
            _wpp.Append(_l3);
            _wpp.Append(_l4);

            //_wp.Append(_wpp);

            Wp.DocProperties _dp = new Wp.DocProperties()
            {
                //Id = Convert.ToUInt32(id),
                Id = Convert.ToUInt32(id),
                //Name = "Picture",
                Name = imageName,
            };

            A.Graphic _g = new A.Graphic();
            A.GraphicData _gd = new A.GraphicData() { Uri = "http://schemas.openxmlformats.org/drawingml/2006/picture" };
            A.Pictures.Picture _pic = new A.Pictures.Picture();

            A.Pictures.NonVisualPictureProperties _nvpp = new A.Pictures.NonVisualPictureProperties();
            A.Pictures.NonVisualDrawingProperties _nvdp = new A.Pictures.NonVisualDrawingProperties()
            {
                Id = Convert.ToUInt32(id),
                //Name = imagePartId
                Name = imageName + ".jpg"
            };
            A.Pictures.NonVisualPictureDrawingProperties _nvpdp = new A.Pictures.NonVisualPictureDrawingProperties();
            _nvpp.Append(_nvdp);
            _nvpp.Append(_nvpdp);


            A.Pictures.BlipFill _bf = new A.Pictures.BlipFill();
            A.Blip _b = new A.Blip()
            {
                Embed = imagePartId,
                CompressionState = A.BlipCompressionValues.Print
            };
            _bf.Append(_b);


            A.Stretch _str = new A.Stretch();
            A.FillRectangle _fr = new A.FillRectangle();
            _str.Append(_fr);
            _bf.Append(_str);

            A.Pictures.ShapeProperties _shp = new A.Pictures.ShapeProperties();
            A.Transform2D _t2d = new A.Transform2D();
            A.Offset _os = new A.Offset()
            {
                X = 0L,
                Y = 0L
            };
            A.Extents _ex = new A.Extents()
            {
                Cx = Width,
                Cy = Height
            };

            _t2d.Append(_os);
            _t2d.Append(_ex);

            A.PresetGeometry _preGeo = new A.PresetGeometry()
            {
                Preset = A.ShapeTypeValues.Rectangle
            };
            A.AdjustValueList _adl = new A.AdjustValueList();
            _preGeo.Append(_adl);

            _shp.Append(_t2d);
            _shp.Append(_preGeo);

            _pic.Append(_nvpp);
            _pic.Append(_bf);
            _pic.Append(_shp);

            _gd.Append(_pic);

            _g.Append(_gd);

            _anchor.Append(_spos);
            _anchor.Append(_hp);
            _anchor.Append(_vp);
            _anchor.Append(_e);
            _anchor.Append(_ee);
            if (warp)
            {
                _anchor.Append(_wp);
            }
            else {
                _anchor.Append(wrapNone);
            }
            _anchor.Append(_dp);
            _anchor.Append(nv);
            _anchor.Append(_g);

            _drawing.Append(_anchor);

            return _drawing;
        }
        //嵌入式图片
        public static void AddImageToBody(Body body, string relationshipId)
        {
            var element =
                 new Drawing(
                     new Wp.Inline(
                         new Wp.Extent() { Cx = 990000L, Cy = 792000L },
                         new Wp.EffectExtent()
                         {
                             LeftEdge = 0L,
                             TopEdge = 0L,
                             RightEdge = 0L,
                             BottomEdge = 0L
                         },
                         new Wp.DocProperties()
                         {
                             Id = (UInt32Value)1U,
                             Name = "Picture 1"
                         },
                         new Wp.NonVisualGraphicFrameDrawingProperties(
                             new A.GraphicFrameLocks() { NoChangeAspect = true }),
                         new A.Graphic(
                             new A.GraphicData(
                                 new PIC.Picture(
                                     new PIC.NonVisualPictureProperties(
                                         new PIC.NonVisualDrawingProperties()
                                         {
                                             Id = (UInt32Value)0U,
                                             Name = "New Bitmap Image.jpg"
                                         },
                                         new PIC.NonVisualPictureDrawingProperties()),
                                     new PIC.BlipFill(
                                         new A.Blip(
                                             new A.BlipExtensionList(
                                                 new A.BlipExtension()
                                                 {
                                                     Uri =
                                                        "{28A0092B-C50C-407E-A947-70E740481C1C}"
                                                 })
                                         )
                                         {
                                             Embed = relationshipId,
                                             CompressionState =
                                             A.BlipCompressionValues.Print
                                         },
                                         new A.Stretch(
                                             new A.FillRectangle())),
                                     new PIC.ShapeProperties(
                                         new A.Transform2D(
                                             new A.Offset() { X = 0L, Y = 0L },
                                             new A.Extents() { Cx = 990000L, Cy = 792000L }),
                                         new A.PresetGeometry(
                                             new A.AdjustValueList()
                                         )
                                         { Preset = A.ShapeTypeValues.Rectangle }))
                             )
                             { Uri = "http://schemas.openxmlformats.org/drawingml/2006/picture" })
                     )
                     {
                         DistanceFromTop = (UInt32Value)0U,
                         DistanceFromBottom = (UInt32Value)0U,
                         DistanceFromLeft = (UInt32Value)0U,
                         DistanceFromRight = (UInt32Value)0U,
                         //EditId = "50D07946"
                     });

            // Append the reference to body, the element should be in a Run.
            body.AppendChild(new Paragraph(new Run(element)));
        }

        public static Drawing CreateInlineDrawing(string relationshipId, List<float> position, int id)
        {
            int scale = 360000;
            Int32 X1 = Convert.ToInt32(position[0] * scale);
            Int32 Y1 = Convert.ToInt32(position[1] * scale);
            UInt32 Width = Convert.ToUInt32(position[2] * scale);
            UInt32 Height = Convert.ToUInt32(position[3] * scale);
            var extent = new Wp.Extent() { Cx = Width, Cy = Height };
            var effectExtent = new Wp.EffectExtent() { LeftEdge = 0L, TopEdge = 0L, RightEdge = 0L, BottomEdge = 0L };
            var docPtr = new Wp.DocProperties() { Name = "picture", Id = Convert.ToUInt32(id)};
            var nvpros = new Wp.NonVisualGraphicFrameDrawingProperties(new A.GraphicFrameLocks() { NoChangeAspect = true });
            var graph = new A.Graphic(
                             new A.GraphicData(
                                 new PIC.Picture(
                                     new PIC.NonVisualPictureProperties(
                                         new PIC.NonVisualDrawingProperties()
                                         {
                                             Id = Convert.ToUInt32(id),
                                             Name = relationshipId
                                         },
                                         new PIC.NonVisualPictureDrawingProperties()),
                                     new PIC.BlipFill(
                                         new A.Blip(
                                             new A.BlipExtensionList(
                                                 new A.BlipExtension()
                                                 {
                                                     Uri = "{28A0092B-C50C-407E-A947-70E740481C1C}"
                                                 })
                                         )
                                         {
                                             Embed = relationshipId,
                                             CompressionState = A.BlipCompressionValues.Print
                                         },
                                         new A.Stretch(
                                             new A.FillRectangle())),
                                     new PIC.ShapeProperties(
                                         new A.Transform2D(
                                             new A.Offset() { X = 0L, Y = 0L },
                                             new A.Extents() { Cx = Width, Cy = Height }),
                                         new A.PresetGeometry(
                                             new A.AdjustValueList()
                                         )
                                         { Preset = A.ShapeTypeValues.Rectangle }))
                             )
                             { Uri = "http://schemas.openxmlformats.org/drawingml/2006/picture" });

            return new Drawing(new Wp.Inline(extent, effectExtent, docPtr, nvpros, graph)
            {
                DistanceFromTop = (UInt32Value)0U,
                DistanceFromBottom = (UInt32Value)0U,
                DistanceFromLeft = (UInt32Value)0U,
                DistanceFromRight = (UInt32Value)0U,
            });
        }

        public static Run GetTabRun() {
            RunProperties runProperties = new RunProperties();
            RunFonts font = new RunFonts();
            font.Hint = FontTypeHintValues.EastAsia;
            runProperties.Append(font);
            //runProperties.Append(new Lang);
            Run r = new Run();
            r.Append(runProperties);
            r.Append(new TabChar());
            return r;
        }
        public static void AddMathFormula(Paragraph paragraph, string ommlString, int fontSize) {
            //var mathPara = paragraph.AppendChild(new Math.Paragraph());
            var mathPara = new Math.Paragraph();
            RunProperties runProperties = new RunProperties();
            var mPr = new Math.ParagraphProperties(new TextAlignment { Val = VerticalTextAlignmentValues.Baseline});
            mathPara.AppendChild(mPr);
            RunFonts font = new RunFonts();
            font.EastAsia = "宋体";
            font.Ascii = "Times New Roman";
            //font.Ascii = "Cambria Math";
            //font.HighAnsi = "Cambria Math";
            font.HighAnsi = "Times New Roman";
            FontSize size = new FontSize();
            //size.Val = new StringValue(((fontSize+5) * 2).ToString());
            runProperties.Append(size);
            runProperties.Append(font);
            Math.OfficeMath formula = new Math.OfficeMath(ommlString);
            //Run r = formula.Elements<Run>().First();
            //RunProperties rPr = GetRunProperties(fontSize);
            //formula.Append(runProperties);
            //mathPara.AppendChild(formula);
            paragraph.AppendChild(formula);
        }
        
    }
    class TableUtils
    {
        public static TableProperties getTableProp()
        {
            TableProperties props = new TableProperties(
               new TableBorders(
               new TopBorder
               {
                   Val = new EnumValue<BorderValues>(BorderValues.Single),
                   Size = 6
               },
               new BottomBorder
               {
                   Val = new EnumValue<BorderValues>(BorderValues.Single),
                   Size = 6
               },
               new LeftBorder
               {
                   Val = new EnumValue<BorderValues>(BorderValues.Single),
                   Size = 6
               },
               new RightBorder
               {
                   Val = new EnumValue<BorderValues>(BorderValues.Single),
                   Size = 6
               },
               new InsideHorizontalBorder
               {
                   Val = new EnumValue<BorderValues>(BorderValues.Single),
                   Size = 6
               },
               new InsideVerticalBorder
               {
                   Val = new EnumValue<BorderValues>(BorderValues.Single),
                   Size = 6
               }));
            Justification justification = new Justification() { Val = JustificationValues.Center };
            TableWidth tw = new TableWidth() { Width = "0", Type=TableWidthUnitValues.Auto};
            TableCellMarginDefault tcm = new TableCellMarginDefault(new LeftMargin(){Width="10",Type=TableWidthUnitValues.Dxa},
                new RightMargin() { Width = "10", Type = TableWidthUnitValues.Dxa });
            TableLook tl = new TableLook() { Val = "0000", FirstRow = false, LastRow = false, FirstColumn = false, LastColumn = false, NoHorizontalBand = false, NoVerticalBand = false };
            TableLayout tableLayout = new TableLayout() { Type=TableLayoutValues.Autofit};
            props.Append(tcm);
            props.Append(tw);
            props.Append(tl);
            props.Append(justification);
            props.Append(tableLayout);
            return props;
        }
        public static TableCellProperties GetTableCellProp(JsonColumIndex[] cell_index, int row_id, int col_id, bool merged) {
            TableCellProperties Tcpr = new TableCellProperties(
                      new TableCellVerticalAlignment { Val = TableVerticalAlignmentValues.Center });
            int[] row_index = cell_index[row_id].row;
            int cell_idx = row_index[col_id];
            if (cell_idx == -1)
            {
                return Tcpr;
            }

            //列方向合并
            if (col_id + 1 < row_index.Length && row_index[col_id+1]==cell_idx) {
                if ((col_id > 0 && row_index[col_id - 1] != cell_idx) || col_id == 0) {
                    HorizontalMerge hMerge = new HorizontalMerge();
                    hMerge.Val = MergedCellValues.Restart;
                    Tcpr.Append(hMerge);
                }
                merged = true;
            }
            if (col_id > 0 && row_index[col_id - 1] == cell_idx)
            {
                HorizontalMerge hMerge = new HorizontalMerge();
                hMerge.Val = MergedCellValues.Continue;
                Tcpr.Append(hMerge);
                merged = true;
            }

            //行方向合并
            if (row_id + 1 < cell_index.Length && cell_index[row_id + 1].row[col_id] == cell_idx)
            {
                //当前单元格与上一行不同格则为合并的开始
                if ((row_id > 0 && cell_index[row_id - 1].row[col_id] != cell_idx) || (row_id == 0))
                {
                    VerticalMerge vMerge = new VerticalMerge();
                    vMerge.Val = MergedCellValues.Restart;
                    Tcpr.Append(vMerge);
                }
                merged = true;
            }

            if (row_id > 0 && cell_index[row_id - 1].row[col_id] == cell_idx)
            {
                VerticalMerge vMerge = new VerticalMerge();
                vMerge.Val = MergedCellValues.Continue;
                Tcpr.Append(vMerge);
                merged = true;
            }

            return Tcpr;
        }

        public static Table AddTable(JsonTable table_meta_data, MainDocumentPart mainPart)
        {
            Table table = new Table();
            TableProperties props = getTableProp();
            table.AppendChild<TableProperties>(props);
            int row_num = table_meta_data.row_num;
            //int row_num = 1;

            int col_num = table_meta_data.col_num;
            //int col_num = 2;
            JsonCell[] cells = table_meta_data.cells;
            for (int i = 0; i < row_num; i++)
            {
                //添加行
                float row_height_cm = table_meta_data.row_heights[i] * table_meta_data.ScaleH;
                UInt32 row_height = Convert.ToUInt32(row_height_cm * 567);
                var tr = new TableRow();
                tr.Append(new TableRowProperties(
                       new TableRowHeight { Val = row_height, HeightType = HeightRuleValues.Exact }));
                tr.Append(new TablePropertyExceptions(new TableCellMarginDefault(new TopMargin() { Width = "0", Type = TableWidthUnitValues.Dxa },
                    new BottomMargin() { Width = "0", Type = TableWidthUnitValues.Dxa }
                    )));
                JsonColumIndex row_index = table_meta_data.cell_index[i];
                int col_index = 0;
                while (col_index < col_num) {
                    int cell_idx = row_index.row[col_index];
                    float cur_width = table_meta_data.col_widths[col_index] * table_meta_data.ScaleW;
                    //ignore cell which merge with last
                    if (col_index > 0 && cell_idx == row_index.row[col_index - 1])
                    {
                        col_index += 1;
                        continue;
                    }
                    else if (cell_idx != -1)
                    {
                        TableCellProperties Tcpr = new TableCellProperties(
                            new TableCellVerticalAlignment { Val = TableVerticalAlignmentValues.Center });
                        int orig_col_index = col_index + 0;
                        //grid span same cell width next colum
                        if (col_index < col_num - 1 && cell_idx == row_index.row[col_index + 1])
                        {
                            col_index = col_index + 1;
                            int grid_span = 2;
                            cur_width += table_meta_data.col_widths[col_index] * table_meta_data.ScaleW;
                            while (col_index < col_num - 1 && cell_idx == row_index.row[col_index + 1])
                            {
                                col_index += 1;
                                grid_span++;
                                cur_width += table_meta_data.col_widths[col_index] * table_meta_data.ScaleW;
                            }
                            Tcpr.Append(new GridSpan() { Val = grid_span });
                        }
                        UInt32 col_width = Convert.ToUInt32(cur_width * 567);
                        Tcpr.Append(new TableCellWidth { Type = TableWidthUnitValues.Dxa, Width = col_width.ToString() });
                        col_index += 1;
                        //行方向合并
                        if (i + 1 < row_num && table_meta_data.cell_index[i + 1].row[orig_col_index] == cell_idx)
                        {
                            //当前单元格与上一行不同格则为合并的开始
                            if ((i > 0 && table_meta_data.cell_index[i - 1].row[orig_col_index] != cell_idx) || (i == 0))
                            {
                                VerticalMerge vMerge = new VerticalMerge();
                                vMerge.Val = MergedCellValues.Restart;
                                Tcpr.Append(vMerge);
                            }

                        }
                        bool ignore_text = false;
                        if (i > 0 && table_meta_data.cell_index[i - 1].row[orig_col_index] == cell_idx)
                        {
                            VerticalMerge vMerge = new VerticalMerge();
                            vMerge.Val = MergedCellValues.Continue;
                            Tcpr.Append(vMerge);
                            ignore_text = true;
                        }
                        JsonPic[] cell_pics = cells[cell_idx].pics;
                        int nPic = cell_pics.Length;
                        var tc = new TableCell();
                        tc.Append(Tcpr);
                  
                        if (cells[cell_idx].paragraphs.Length == 0 || ignore_text)
                        {
                            if (nPic == 0)
                            {
                                tc.Append(new Paragraph());
                            }
                            else {
                                WordDocBuilder.AddParagraph(tc, null, cell_pics, mainPart);
                            }
                            
                        }
                        else {
   
                            for (int para_id = 0; para_id < cells[cell_idx].paragraphs.Length; para_id++)
                            {
                                WordDocBuilder.AddParagraph(tc, cells[cell_idx].paragraphs[para_id], cell_pics, mainPart);
                            }                            
                        }
                        tr.Append(tc);
                    }
                    else
                    {
                        TableCellProperties Tcpr = new TableCellProperties(
                            new TableCellVerticalAlignment { Val = TableVerticalAlignmentValues.Center });
                        UInt32 col_width = Convert.ToUInt32(cur_width * 567);
                        Tcpr.Append(new TableCellWidth { Type = TableWidthUnitValues.Dxa, Width = col_width.ToString() });
                        var tc = new TableCell();
                        tc.Append(new Paragraph());
                        tc.Append(Tcpr);
                        tr.Append(tc);
                        col_index += 1;
                    }
                }
                table.Append(tr);
            }
            return table;
        }
    }
}
