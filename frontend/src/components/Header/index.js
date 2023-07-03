import React from 'react';
import { Link } from 'react-router-dom';
import './Header.css';

const Header = () => {
  return (
    <nav className="nav">
      <ul className="ul">
        <li className="li">
          <Link to="/" className="link">Home</Link>
        </li>
        <li className="li">
          <Link to="/upload" className="link">Upload</Link>
        </li>
        <li className="li">
          <Link to="/browse" className="link">Browse</Link>
        </li>
      </ul>
    </nav>
  );
};

export default Header;
