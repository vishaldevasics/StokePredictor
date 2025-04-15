import React from "react";
import OpenAI from "openai";
import StrokeReport from "./StrokeReport";

const GptChat = ({ formData, response }) => {
  const [report, setReport] = React.useState("");
  const [isLoading, setIsLoading] = React.useState(false);

  const handlegptClick = async () => {
    setIsLoading(true);
    const openai = new OpenAI({
      apiKey: import.meta.env.VITE_OPENAI_API_KEY,
      dangerouslyAllowBrowser: true,
    });

    const chatCompletion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "user",
          content: `Based on the provided patient information: ${JSON.stringify(
            formData
          )}, and the preliminary evaluation from the machine learning model: ${response}, please generate a detailed stroke risk assessment report. As a virtual health assistant, interpret the data like a medical professional and offer clear, personalized health advice and preventive recommendations to support better cardiovascular and neurological health outcomes.`,
        },
      ],
    });

    setReport(chatCompletion.choices[0].message.content);
    setIsLoading(false);
  };

  return (
    <div className="h-full flex flex-col">
     {report.length === 0 && <button
        className="bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 text-white font-semibold py-2 px-4 rounded-lg shadow-md mb-4 transition-transform hover:scale-105"
        onClick={handlegptClick}
      >
        Check for Result 🧠
      </button>}

      {isLoading && <div className="text-orange-600 font-semibold text-center">Generating report...</div>}

      {!isLoading && report && (
        <div className="bg-white p-4 rounded-lg shadow-inner overflow-y-auto  h-full border border-orange-200">
          <pre className=" whitespace-pre-wrap text-sm text-orange-800">{report}</pre>
          <StrokeReport report={report} />
        </div>
      )}
    </div>
  );
};

export default GptChat;
