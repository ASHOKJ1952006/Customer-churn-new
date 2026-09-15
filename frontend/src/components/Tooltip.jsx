import React, { useState, useRef, useEffect } from 'react';
import './Tooltip.css';

const Tooltip = ({ children, content }) => {
  const [isVisible, setIsVisible] = useState(false);
  const tooltipRef = useRef(null);
  const buttonRef = useRef(null);

  const toggleTooltip = () => {
    setIsVisible(!isVisible);
  };

  const handleClickOutside = (event) => {
    if (tooltipRef.current && !tooltipRef.current.contains(event.target) &&
        buttonRef.current && !buttonRef.current.contains(event.target)) {
      setIsVisible(false);
    }
  };

  useEffect(() => {
    if (isVisible) {
      document.addEventListener('mousedown', handleClickOutside);
    } else {
      document.removeEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isVisible]);

  return (
    <span className="tooltip-wrapper">
      <button
        ref={buttonRef}
        className="tooltip-icon"
        onClick={toggleTooltip}
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        aria-label="Information"
        aria-describedby="tooltip-content"
      >
        ?
      </button>
      {isVisible && (
        <div
          ref={tooltipRef}
          id="tooltip-content"
          className="tooltip-content"
          role="tooltip"
        >
          {content}
        </div>
      )}
      {children}
    </span>
  );
};

export default Tooltip;
