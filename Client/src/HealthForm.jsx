import React, { useState, useEffect } from "react";
import axios from "axios";
import GptChat from "./GptChat";

const HealthForm = () => {
  const [userId, setUserId] = useState(null);

  useEffect(() => {
    const id = localStorage.getItem("userId");
    setUserId(id);
  }, []);

  const [formData, setFormData] = useState({
    gender: "",
    age: "",
    hypertension: "",
    heartDisease: "",
    everMarried: "",
    workType: "",
    residenceType: "",
    avgGlucoseLevel: "",
    bmi: "",
    smokingStatus: "",
  });

  const [result, setResult] = useState("Fill Details to Check");

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === "checkbox" ? checked : value,
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (Object.values(formData).some((val) => val === "")) {
      alert("Fill all details");
      return;
    }

    axios
      .post("http://127.0.0.1:5000/predict", { ...formData, userId })
      .then((response) => {
        setResult(response.data === 0 ? "Low Chance" : "High Chance");
      })
      .catch((error) => console.log(error));
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      {/* Left Section - Form (60%) */}
      <div className="w-[50%] h-screen bg-white bg-opacity-20 backdrop-blur-lg border-r border-orange-200 flex items-center justify-center">
        <div className="w-full max-w-lg h-[90vh] overflow-y-auto p-6 bg-white bg-opacity-30 rounded-2xl shadow-xl">
          <form onSubmit={handleSubmit} className="space-y-4">
            <h2 className="text-3xl font-bold text-orange-600 mb-4 text-center">
              🧠 Stroke Risk Predictor
            </h2>

            {/* Dropdown Inputs */}
            {[
              {
                label: "Gender",
                name: "gender",
                options: [
                  { val: 0, label: "Male" },
                  { val: 1, label: "Female" },
                ],
              },
              {
                label: "Hypertension",
                name: "hypertension",
                options: [
                  { val: 1, label: "Yes" },
                  { val: 0, label: "No" },
                ],
              },
              {
                label: "Heart Disease",
                name: "heartDisease",
                options: [
                  { val: 1, label: "Yes" },
                  { val: 0, label: "No" },
                ],
              },
              {
                label: "Ever Married",
                name: "everMarried",
                options: [
                  { val: 1, label: "Yes" },
                  { val: 0, label: "No" },
                ],
              },
              {
                label: "Work Type",
                name: "workType",
                options: [
                  { val: 2, label: "Private" },
                  { val: 3, label: "Self Employed" },
                  { val: 0, label: "Government Job" },
                  { val: 4, label: "Children" },
                  { val: 1, label: "Unemployed" },
                ],
              },
              {
                label: "Residence Type",
                name: "residenceType",
                options: [
                  { val: 0, label: "Rural" },
                  { val: 1, label: "Urban" },
                ],
              },
              {
                label: "Smoking Status",
                name: "smokingStatus",
                options: [
                  { val: 2, label: "Never Smoked" },
                  { val: 1, label: "Formerly Smoked" },
                  { val: 3, label: "Smokes" },
                  { val: 0, label: "Don't Know" },
                ],
              },
            ].map((field) => (
              <div key={field.name}>
                <label className="block text-orange-700 font-medium mb-1">
                  {field.label}
                </label>
                <select
                  name={field.name}
                  value={formData[field.name]}
                  onChange={handleChange}
                  className="w-full px-4 py-2 rounded-lg bg-white bg-opacity-70 border border-orange-300 focus:outline-none focus:ring-2 focus:ring-orange-400 text-orange-800"
                >
                  <option value="">Select</option>
                  {field.options.map((opt) => (
                    <option key={opt.val} value={opt.val}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
            ))}

            {/* Number Inputs */}
            {[
              { name: "age", label: "Age" },
              { name: "avgGlucoseLevel", label: "Average Glucose Level" },
              { name: "bmi", label: "BMI" },
            ].map((input) => (
              <div key={input.name}>
                <label className="block text-orange-700 font-medium mb-1">
                  {input.label}
                </label>
                <input
                  type="number"
                  name={input.name}
                  value={formData[input.name]}
                  onChange={handleChange}
                  className="w-full px-4 py-2 rounded-lg bg-white bg-opacity-70 border border-orange-300 focus:outline-none focus:ring-2 focus:ring-orange-400 text-orange-800"
                />
              </div>
            ))}

            <button
              type="submit"
              className="w-full py-3 px-4 bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 text-white font-semibold rounded-lg shadow-md transition duration-300 transform hover:scale-105"
            >
              Submit
            </button>

            <div className="mt-4 text-center text-lg font-semibold">
              <span
                className={
                  result === "Low Chance" ? "text-green-600" : "text-red-600"
                }
              >
                Risk of Stroke: {result}
              </span>
            </div>
          </form>
        </div>
      </div>

      {/* Right Section - GPT Report (40%) */}
      <div className="w-[50%] h-screen bg-orange-50 flex items-center justify-center p-6">
        <div className="w-full max-w h-full overflow-y-auto p-4 bg-white bg-opacity-70 rounded-lg shadow-md">
          <GptChat formData={formData} response={result} />
        </div>
      </div>
    </div>
  );
};

export default HealthForm;
