import React, { useState } from "react";
import { Bar } from "react-chartjs-2";
import axios from "axios";

export default function UploadChart() {
  const [chartData, setChartData] = useState(null);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://localhost:5000/upload-analyze", formData);
      const data = res.data;

      setChartData({
        labels: ["Positive", "Negative"],
        datasets: [
          {
            label: "Sentiment count",
            data: [data.positive, data.negative],
            backgroundColor: ["#4caf50", "#f44336"]
          }
        ]
      });
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div>
      <input type="file" onChange={handleUpload} />
      {chartData && <Bar data={chartData} />}
    </div>
  );
}
