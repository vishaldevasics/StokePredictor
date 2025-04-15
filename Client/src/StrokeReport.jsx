import React, { useRef } from "react";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";

const StrokeReport = ({ report }) => {
  const reportRef = useRef();
  const downloadCode = () => {
    const doc = new jsPDF();
    let x = 10;
    let y = 10;
    const lineHeight = 10;
    const pageWidth = doc.internal.pageSize.getWidth();
    const margin = 10;
    const maxLineWidth = pageWidth - margin * 2;
  
    // Split the report into separate lines by newline character.
    const lines = report.split("\n").filter((line) => line.trim() !== "");
  
    lines.forEach((rawLine) => {
      const line = rawLine.trim();
  
      // Check for a horizontal rule (e.g., ---)
      if (line.startsWith("---")) {
        doc.setLineWidth(0.5);
        doc.line(x, y, pageWidth - margin, y);
        y += lineHeight * 0.5;
        return;
      }
  
      // Heading with ** … ** and not a field label
      if (line.startsWith("**") && line.endsWith("**") && !line.includes(":")) {
        doc.setFont("helvetica", "bold");
        doc.setFontSize(18);
        const text = line.replace(/\*\*/g, "");
        const splitted = doc.splitTextToSize(text, maxLineWidth);
        doc.text(splitted, x, y);
        y += splitted.length * lineHeight;
        return;
      }
  
      // Bold section titles or labels (assumed if enclosed in **…**)
      if (line.startsWith("**") && line.endsWith("**")) {
        doc.setFont("helvetica", "bold");
        doc.setFontSize(14);
        const text = line.replace(/\*\*/g, "");
        const splitted = doc.splitTextToSize(text, maxLineWidth);
        doc.text(splitted, x, y);
        y += splitted.length * lineHeight;
        return;
      }
  
      // Markdown header level 3 (e.g., ### …)
      if (line.startsWith("###")) {
        doc.setFont("helvetica", "bold");
        doc.setFontSize(16);
        const text = line.replace(/###\s*/, "");
        const splitted = doc.splitTextToSize(text, maxLineWidth);
        doc.text(splitted, x, y);
        y += splitted.length * lineHeight;
        return;
      }
  
      // Bullet points (lines that start with '-' or '- ')
      if (line.startsWith("-")) {
        doc.setFont("helvetica", "normal");
        doc.setFontSize(12);
        // Indent bullet items
        const bulletX = x + 5;
        const text = line;
        const splitted = doc.splitTextToSize(text, maxLineWidth - 5);
        doc.text(splitted, bulletX, y);
        y += splitted.length * lineHeight;
        return;
      }
  
      // Otherwise treat as normal paragraph text
      doc.setFont("helvetica", "normal");
      doc.setFontSize(12);
      const splitted = doc.splitTextToSize(line, maxLineWidth);
      doc.text(splitted, x, y);
      y += splitted.length * lineHeight;
  
      // Add space between paragraphs if needed
      y += 2;
    });
  
    doc.save("code.pdf");
  };
  
  if (!report) return null;
  return (
    <div className="w-full flex flex-col items-center mt-8 px-4">
      <div
        ref={reportRef}
        className="bg-white text-black shadow-md p-6 rounded-2xl max-w-3xl w-full font-sans leading-relaxed"
        style={{ fontSize: "1rem", lineHeight: "1.75rem" }}
      >
        <h2 className="text-2xl font-bold mb-4 text-center">
          🧠 Stroke Risk Assessment Report
        </h2>
        <div dangerouslySetInnerHTML={{ __html: report }} />
      </div>

      <button
        onClick={downloadCode}
        className="mt-6 px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow hover:bg-blue-700 transition duration-200"
      >
        📄 Download Report as PDF
      </button>
    </div>
  );
};

export default StrokeReport;
