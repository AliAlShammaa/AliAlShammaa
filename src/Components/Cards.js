import React from "react";

function Cards(props) {
  const ColoredLine = ({ color }) => (
    <hr
      style={{
        color: color,
        backgroundColor: color,
        height: 5,
      }}
    />
  );

  return (
    <div className="col-md-4 mt-4 pt-4">
      <div className="card bg-dark border border-light">
        <a
          className="btn-outline-dark btn btn-light"
          href={props.link}
          target="blank"
        >
          {" "}
          <img className="card-img-top" src={props.src}></img>{" "}
        </a>
        <div className="card-body ">
          <em>
            {" "}
            <h1 className="card-title text-light"> {props.title}</h1>{" "}
          </em>
          <p className="card-text text-light"> {props.txt}</p>
        </div>
      </div>
    </div>
  );
}

export default Cards;
