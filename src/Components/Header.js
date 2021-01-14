import { React, useEffect } from "react";
import { Link } from "react-router-dom";
import Logo from "../Images/aa-icon-272.jpg";
import "../styles/Header.css";

function Header(props) {
  window.onscroll = function () {
    unStick();
  };
  let up = false;
  let home = true;

  let change = function () {
    let toggle = document.getElementById("toggle");
    if (!up) {
      toggle.classList.remove("fa-toggle-down");
      toggle.classList.add("fa-toggle-up");
      up = true;
    } else {
      up = false;
      toggle.classList.add("fa-toggle-down");
      toggle.classList.remove("fa-toggle-up");
    }
  };

  let switchActive = function () {
    let homeItem = document.getElementById("navbar-home");
    let MssgMe = document.getElementById("navbar-msg");

    if (home) {
      homeItem.classList.remove("active");
      MssgMe.classList.add("active");
      home = false;
    } else {
      homeItem.classList.add("active");
      MssgMe.classList.remove("active");
      home = true;
    }
  };

  let unStick = function () {
    let navbar = document.getElementById("header");
    let rule = document.getElementById("rule1");
    console.log("inside unstick");
    if (rule.offsetTop <= window.pageYOffset) {
      navbar.classList.remove("sticky-top");
    } else {
      navbar.classList.add("sticky-top");
    }
  };

  return (
    <div className="sticky-top m-0" id="header">
      <nav
        id="navbar"
        className="navbar navbar-expand-md navbar-light collapse bg-light"
      >
        <div className="container-fluid">
          <span className="navbar-brand ml-2 ">
            <span className="mb-4">
              <img src={Logo} className="logo rounded" />
            </span>
            <span
              className="text-success btn-outline-dark rounded p-2 m-2 brand"
              href="#"
            >
              Ali Al Shammaa
            </span>
          </span>
          {/* <button
            className="navbar-toggler"
            type="button"
            data-toggle="collapse"
            data-target="#navbarResponsive"
          >
            <span className="navbar-toggler-icon"></span>{" "}
          </button> */}
          <div id="navbarResponsive">
            <ul className="ml-auto navbar-nav">
              <li
                id="navbar-home"
                className="nav-item active"
                onClick={switchActive}
              >
                {" "}
                <Link to="/" className="nav-link">
                  Home
                </Link>
              </li>

              <li id="navbar-msg" className="nav-item" onClick={switchActive}>
                <Link to="/MessageAli" className="nav-link">
                  Message
                </Link>
              </li>
            </ul>
          </div>
        </div>
      </nav>
      <button
        id="toggle-button"
        className="Navigate btn-block text-dark btn btn-light btn-outline-dark btn-sm"
        data-toggle="collapse"
        data-target="#navbar"
        onClick={change}
      >
        <i id="toggle" className="fa fa-toggle-down"></i>
      </button>
      {/* <p> */}
      {/* Keep to stay under the sticky-top*/}
      {/* <br />
        <br />
      </p> */}
    </div>
  );
}

export default Header;
