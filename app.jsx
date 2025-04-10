// App.jsx
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { FarmerTable } from './components/FarmerTable';
import { MarketTable } from './components/MarketTable';

export default function App() {
  const [farmers, setFarmers] = useState([]);
  const [markets, setMarkets] = useState([]);

  useEffect(() => {
    axios.get('/api/farmers').then(res => setFarmers(res.data));
    axios.get('/api/markets').then(res => setMarkets(res.data));
  }, []);

  return (
    <div className="p-6 font-sans bg-gray-100 min-h-screen">
      <h1 className="text-3xl font-bold mb-6 text-center">AgriZen Dashboard</h1>
      <div className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Farmer Information</h2>
        <FarmerTable data={farmers} />
      </div>
      <div>
        <h2 className="text-2xl font-semibold mb-4">Market Research</h2>
        <MarketTable data={markets} />
      </div>
    </div>
  );
}
