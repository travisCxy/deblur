using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
namespace imageToWord
{

    class program
    {

        static void Main(string[] args)
        {
            while (true)
            {
                string input = Console.ReadLine();
                string[] paths = input.Split(';');
                string jsonPath = paths[0];
                string docPath = paths[1];
                string pid = paths[2];
                string output = "success";
                try
                {
                    WordDocBuilder.CreateDoc1(jsonPath, docPath);
                }
                catch (Exception)
                {
                    output = "failed";
                    Console.WriteLine(output);
                    continue;
                }

                // 将输出写入标准输出流
                Console.WriteLine(output);
            }
        }

        //static void Main(string[] args)
        //{
        //    bool debug = true;
        //    bool debug = false;
        //    string jsonPath = "workdir\\a.json";
        //    string docPath = "workdir\\a.docx";
        //    if (!debug)
        //    {
        //        jsonPath = args[0];
        //        docPath = args[1];
        //    }
        //    WordDocBuilder.CreateDoc1(jsonPath, docPath);

        //    string inputDoc = "workdir\\test.docx";
        //    WordDocBuilder.ConvertDocx(inputDoc, inputDoc);
        //}
    }
}
