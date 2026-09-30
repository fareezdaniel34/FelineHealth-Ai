import { useEffect, useState } from 'react';
import './App.css';

function App() {
  const [message, setMessage] = useState('Loading...');

  useEffect(() => {
    fetch('http://127.0.0.1:5000/api/hello')
      .then((res) => res.json())
      .then((data) => setMessage(data.message))
      .catch((err) => console.error('Error connecting to backend:', err));
  }, []);

  return (
    <main style={{ textAlign: 'center', marginTop: '4rem', fontFamily: 'sans-serif' }}>
      <h1>React + Flask Setup</h1>
      <p>Backend response: <strong>{message}</strong></p>
    </main>
  );
}

export default App;