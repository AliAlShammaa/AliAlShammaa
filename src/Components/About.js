import React from "react";
import "../styles/About.css";

export default function About() {
  return (
    <div className="about col-12 container-fluid">
      <div className="row ">
        <div className="col-11 col-md-9 col-lg-8 ml-4 mt-4">
          <h1>About me</h1>
          <div className="mt-4 ml-3">
            <p>
              I am a 1st year Computer Science student at UWaterloo in Ontario,
              Canada. I am passionate about problem-solving and ready to take on
              any challenge that tests my skills in computer science, math and
              more...
            </p>
            <br />
            <p>
              I recently finished my 1A term with 93% Cumulative Avg and 94%
              Math Avg. I look forward to learning more CS and Math at UW.
            </p>
            <br />
            <p>
              I recently finnished an internship at Atlas365 where I worked on
              developing the Front-end of their cryptocurrency's website called
              <a
                className="btn-outline-info"
                href="https://atlas365.ca/ecopoints"
              >
                {" "}
                EcoPoints.{" "}
              </a>{" "}
              I also developed social media platform bots for a gaming studio
              called GamePill.
            </p>
            <br />
            <p>
              In the future, I see myself researching new technologies from
              Artificial Intelligence to Quantum Computers. In my free time, I
              do Web development, bot automation and sometimes (hehehe not
              always 😂) workout!!
            </p>
            <br />
            <p></p>
          </div>
        </div>

        <div className="col-4">{/* <History /> */}</div>
        {/* <div className="col-4 pl-5 ml-5 mr-0 pr-0"></div> */}
      </div>
    </div>
  );
}
