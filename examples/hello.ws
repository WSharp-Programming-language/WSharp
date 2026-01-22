// Hello World example in W#

fn main() {
    let message = "Hello, W#!";
    print(message);
}

// Variable declarations with type inference and explicit types
fn variables_demo() {
    let x = 42;                    // Inferred as i32
    let y: i64 = 100;              // Explicit type
    let mut counter = 0;           // Mutable variable
    counter = counter + 1;

    let name: String = "W#";
    let pi = 3.14159;              // Inferred as f64
    let is_awesome = true;         // Inferred as bool
}

// Function with parameters and return type
fn add(a: i32, b: i32) -> i32 {
    a + b  // Implicit return (last expression)
}

// Async function
async fn fetch_data(url: String) -> String {
    let response = await http_get(url);
    response.body
}

// HTTP status handling with first-class types
fn handle_response(status: http 200) -> String {
    "Success!"
}

fn handle_response(status: http 404) -> String {
    "Not Found"
}

fn handle_response(status: http 5xx) -> String {
    "Server Error"
}

// Match expression with HTTP status patterns
fn process_status(code: http) -> String {
    match code {
        http 200 => "OK",
        http 201 => "Created",
        http 2xx => "Success",
        http 404 => "Not Found",
        http 4xx => "Client Error",
        http 5xx => "Server Error",
        _ => "Unknown",
    }
}

// Prototype (class-like) definition
proto Person {
    name: String
    age: i32

    fn new(name: String, age: i32) {
        self.name = name;
        self.age = age;
    }

    fn greet(self) -> String {
        "Hello, my name is " + self.name
    }

    fn birthday(self) {
        self.age = self.age + 1;
    }
}

// Extending a prototype with new methods
extend Person {
    fn age_in_days(self) -> i32 {
        self.age * 365
    }
}

// Object literals
fn create_point() {
    let point = {
        x: 10,
        y: 20,
    };

    let distance = (point.x * point.x + point.y * point.y);
}

// Control flow
fn control_flow_demo(n: i32) -> i32 {
    // If expression
    let result = if n > 0 {
        n * 2
    } else {
        0
    };

    // While loop
    let mut i = 0;
    while i < n {
        i = i + 1;
    }

    // For loop
    for x in 0..10 {
        print(x);
    }

    // Loop with break
    let mut count = 0;
    loop {
        count = count + 1;
        if count >= 5 {
            break count
        }
    }
}

// Lambda expressions
fn higher_order_demo() {
    let double = fn(x: i32) -> i32 => x * 2;
    let result = double(21);

    // Async lambda
    let fetch = async fn(url: String) -> String {
        await http_get(url)
    };
}

// Multiple dispatch example
fn process(data: String) -> String {
    "Processing string: " + data
}

fn process(data: i32) -> String {
    "Processing integer"
}

fn process(data: [i32]) -> String {
    "Processing array of integers"
}

// Generics
fn identity<T>(value: T) -> T {
    value
}

// Type aliases
type StringList = [String];
type Handler = fn(String) -> String;
type AsyncHandler = async fn(String) -> String;

// Modules
mod utils {
    pub fn helper() -> i32 {
        42
    }
}

// Imports
use std::io;
use std::collections::{Vec, Map};
