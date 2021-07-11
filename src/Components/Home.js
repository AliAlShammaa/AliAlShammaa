import React, { useEffect, useState } from "react";
import Intro from "./Intro.js";
import About from "./About.js";
import Project from "./Projects.js";
import History from "./inHistory";
import Resume from "../files/Ali AlShammaa-Resume.pdf";
import "../styles/Home.css";

export default function Home() {
  // let [bool, setBool] = useState(true);
  // useEffect(() => {});

  return (
    <div className="col-12 container-fluid pr-0 pl-0 ">
      <div id="bgPic">
        {/*
        
        */}
        <div id="bgPic2">
          <h1 id="siteTitle" className="hide">
            <span> Ali Al Shammaa</span>
          </h1>
          <div id="homeDown" className="bounce">
            <a href="#home">
              <i class="fa fa-chevron-down"></i>
            </a>
          </div>
          <div id="socialLinks">
            <a
              id="linkedin"
              className="mr-2"
              href="https://github.com/AliAlShammaa"
              target="blank"
            >
              <i class="fa fa-github" aria-hidden="true"></i>{" "}
            </a>
            <a
              id="Github"
              className="mr-2 ml-2"
              href="https://www.linkedin.com/in/ali-al-shammaa-508a59bb/"
              target="blank"
            >
              <i class="fa fa-linkedin" aria-hidden="true"></i>{" "}
            </a>
            <a id="Resume" className="ml-2" href={Resume} target="blank">
              <i class="fa fa-file-pdf-o" aria-hidden="true"></i>{" "}
            </a>
          </div>
        </div>
      </div>
      <div id="home" className="pt-4">
        <section className="mt-4 pt-2">
          <Intro />
          <br />
          <hr id="rule1" className="home-hr" />
          <About />
          <hr className="home-hr" />
          <History />
          <hr className="home-hr" />
          <Project />
          <br />
          <hr className="home-hr" />
        </section>
      </div>
    </div>
  );
}
