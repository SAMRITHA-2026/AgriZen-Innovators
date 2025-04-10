// Example hardcoded data - replace with fetch('/api/...') if needed
document.getElementById("recommended-crops").innerHTML = `
  <li>Wheat</li>
  <li>Rice</li>
`;

document.querySelector("#trending-crops tbody").innerHTML = `
  <tr><td>Corn</td><td>199.99</td><td>$454.79</td></tr>
  <tr><td>Soybean</td><td>199.98</td><td>$141.79</td></tr>
`;

document.getElementById("sustainable-crops").innerHTML = `
  <li>Wheat (Yield: 8.87, Sustainability: 89.76)</li>
  <li>Soybean (Yield: 6.22, Sustainability: 82.92)</li>
`;

document.getElementById("sustainability-score").textContent = "50.80";
document.getElementById("predicted-yield").textContent = "5.50 tons per acre";

// script.js

async function fetchData() {
    // Recommended Crops
    const rec = await fetch("/api/recommendations").then(res => res.json());
    document.getElementById("recommended-crops").innerHTML =
      rec.map(crop => `<li>${crop}</li>`).join("");
  
    // Trending Crops
    const trend = await fetch("/api/trending").then(res => res.json());
    document.querySelector("#trending-crops tbody").innerHTML =
      trend.map(item => `
        <tr>
          <td>${item.Product}</td>
          <td>${item.Demand_Index}</td>
          <td>$${item.Market_Price_per_ton}</td>
        </tr>
      `).join("");
  
    // Sustainable Crops
    const sustainable = await fetch("/api/sustainable").then(res => res.json());
    document.getElementById("sustainable-crops").innerHTML =
      sustainable.map(c => `<li>${c.Crop_Type} (Yield: ${c.Crop_Yield_ton}, Sustainability: ${c.Sustainability_Score})</li>`).join("");
  
    // Prediction Summary
    const pred = await fetch("/api/predictions").then(res => res.json());
    document.getElementById("sustainability-score").textContent = pred.sustainability_score;
    document.getElementById("predicted-yield").textContent = `${pred.predicted_yield} tons per acre`;
  }
  
  // Call function on page load
  fetchData();
  