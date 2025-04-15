import React, { useState,useEffect } from "react";
import axios from "axios";
import "./App.css";
import GptChat from "./GptChat";
import StrokeReport from "./StrokeReport";
const HealthForm = () => {
  const [userId, setUserId] = useState(null);

  useEffect(() => {
    const id = localStorage.getItem('userId');
    setUserId(id);
  }, [userId]);
  // console.log(userId);
  
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
    if (
      formData.gender === "" ||
      formData.age === "" ||
      formData.hypertension === "" ||
      formData.everMarried === "" ||
      formData.workType === "" ||
      formData.residenceType === "" ||
      formData.avgGlucoseLevel === "" ||
      formData.bmi === "" ||
      formData.smokingStatus === "" ||
      formData.heartDisease === ""
    ) {
      alert("Fill all details");
      return;
    }
    // console.log(formData)
    axios
  .post("http://127.0.0.1:5000/predict", {...formData, userId: userId})
  .then((response) => {
    console.log(response);
    if (response.data === 0) {
      setResult("Low Chance");
    } else {
      setResult("High Chance");
    }
  })
  .catch((error) => {
    console.log(error);
  });

  };

  return (
    <>
      <form onSubmit={handleSubmit}>
        <div className="headsub">Predict Your Risk of Stroke Today!</div>
        <label>
          Gender:
          <select name="gender" value={formData.gender} onChange={handleChange}>
            <option value="">Select</option>
            <option value={0}>Male</option>
            <option value={1}>Female</option>
          </select>
        </label>
        <label>
          Age:
          <input
            type="number"
            name="age"
            value={formData.age}
            onChange={handleChange}
          />
        </label>
        <label>
          Hypertension:
          <select
            name="hypertension"
            value={formData.hypertension}
            onChange={handleChange}
          >
            <option value="">Select</option>
            <option value={1}>Yes</option>
            <option value={0}>No</option>
          </select>
        </label>
        <label>
          Heart Disease:
          <select
            name="heartDisease"
            value={formData.heartDisease}
            onChange={handleChange}
          >
            <option value="">Select</option>
            <option value={1}>Yes</option>
            <option value={0}>No</option>
          </select>
        </label>
        <label>
          Ever Married:
          <select
            name="everMarried"
            value={formData.everMarried}
            onChange={handleChange}
          >
            <option value="">Select</option>
            <option value={1}>Yes</option>
            <option value={0}>No</option>
          </select>
        </label>
        <label>
          Work Type:
          <select
            name="workType"
            value={formData.workType}
            onChange={handleChange}
          >
            <option value="">Select</option>
            <option value={2}>Private</option>
            <option value={3}>Self Employed</option>
            <option value={0}>Government Job</option>
            <option value={4}>Children</option>
            <option value={1}>Unemployed</option>
          </select>
        </label>
        <label>
          Residence Type:
          <select
            name="residenceType"
            value={formData.residenceType}
            onChange={handleChange}
          >
            <option value="">Select</option>
            <option value={0}>Rural</option>
            <option value={1}>Urban</option>
          </select>
        </label>
        <label>
          Average Glucose Level:
          <input
            type="number"
            name="avgGlucoseLevel"
            value={formData.avgGlucoseLevel}
            onChange={handleChange}
          />
        </label>
        <label>
          BMI:
          <input
            type="number"
            name="bmi"
            value={formData.bmi}
            onChange={handleChange}
          />
        </label>
        <label>
          Smoking Status:
          <select
            name="smokingStatus"
            value={formData.smokingStatus}
            onChange={handleChange}
          >
            <option value="">Select</option>
            <option value={2}>Never Smoked</option>
            <option value={1}>Formerly Smoked</option>
            <option value={3}>Smokes</option>
            <option value={0}>Don't Know about smoking</option>
          </select>
        </label>
        <button type="submit">Submit</button>
      </form>
      <div>
        <GptChat formData={formData} response={result} />
        Risk of Stroke :{" "}
        <span className={result === "Low Chance" ? "low" : "high"}>
          {result}
        </span>
      </div>
    </>
  );
};

export default HealthForm;
