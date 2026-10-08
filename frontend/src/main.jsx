import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import PawBackground from "./PawBackground.jsx";

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <PawBackground />
    <App />
  </StrictMode>,
)
