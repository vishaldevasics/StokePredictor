import React, { useRef } from "react";
import jsPDF from "jspdf";

const StrokeReport = ({ report }) => {
  const reportRef = useRef();
  const downloadCode = () => {
    const doc = new jsPDF();
    let x = 10;
    let y = 10;
    const lineHeight = 10;
    const margin = 10;
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const maxLineWidth = pageWidth - margin * 2;
  
    const checkPageSpace = (linesNeeded) => {
      if (y + linesNeeded * lineHeight > pageHeight - margin) {
        doc.addPage();
        y = margin;
      }
    };
  
    const addText = (text, font = "normal", size = 12, indent = 0) => {
      doc.setFont("helvetica", font);
      doc.setFontSize(size);
      const splitted = doc.splitTextToSize(text, maxLineWidth - indent);
      checkPageSpace(splitted.length);
      doc.text(splitted, x + indent, y);
      y += splitted.length * lineHeight + 2;
    };
  
    const addHorizontalLine = () => {
      checkPageSpace(1);
      doc.setLineWidth(0.5);
      doc.line(x, y, pageWidth - margin, y);
      y += lineHeight * 0.5;
    };
  
    const lines = report.split("\n").filter((line) => line.trim() !== "");
  
    lines.forEach((rawLine) => {
      const line = rawLine.trim();
  
      if (line.startsWith("---")) {
        addHorizontalLine();
        return;
      }
  
      // Heading (** … **) without colon
      if (line.startsWith("**") && line.endsWith("**") && !line.includes(":")) {
        addText(line.replace(/\*\*/g, ""), "bold", 18);
        return;
      }
  
      // Bold section titles or labels (**…**)
      if (line.startsWith("**") && line.endsWith("**")) {
        addText(line.replace(/\*\*/g, ""), "bold", 14);
        return;
      }
  
      // Markdown header level 3 (### …)
      if (line.startsWith("###")) {
        addText(line.replace(/###\s*/, ""), "bold", 16);
        return;
      }
  
      // Bullet points (- …)
      if (line.startsWith("-")) {
        addText(line, "normal", 12, 5);
        return;
      }
  
      // Normal paragraph
      addText(line, "normal", 12);
    });
  
    doc.save("stroke-report.pdf");
  };

  if (!report) return null;
  return (
    <button
      onClick={downloadCode}
      className="mt-4 w-full bg-orange-500 hover:bg-orange-600 text-white font-semibold py-2 px-4 rounded-lg shadow transition-transform hover:scale-105"
    >
      📄 Download Report as PDF
    </button>
  );
};

export default StrokeReport;
