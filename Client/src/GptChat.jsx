import React from "react";
import OpenAI from "openai";
import StrokeReport from "./StrokeReport";

const gptChat = ({ formData, response }) => {
  const [report, setReport] = React.useState([]);

  const handlegptClick = async () => {
    const openai = new OpenAI({
      apiKey: import.meta.env.VITE_OPENAI_API_KEY,
      dangerouslyAllowBrowser: true,
    });

    const chatCompletion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "user",
          content: `Based on the provided patient information: ${JSON.stringify(formData)}, and the preliminary evaluation from the machine learning model: ${response}, please generate a detailed stroke risk assessment report. As a virtual health assistant, interpret the data like a medical professional and offer clear, personalized health advice and preventive recommendations to support better cardiovascular and neurological health outcomes.`,
        },
      ],
    });

    setReport(chatCompletion.choices[0].message.content);
  };

  React.useEffect(() => {
    console.log(report);
  }, [report]);

  return (
    <div>
      <button className="btn_chat" onClick={handlegptClick}>
        Check for Result
      </button>
      <button><StrokeReport report={report} /></button>
    </div>
  );
};

export default gptChat;
